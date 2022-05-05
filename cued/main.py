from itertools import product
import numpy as np
from numpy.fft import fftshift, fft, ifftshift, ifft, fftfreq
from scipy.integrate import ode
import time
from typing import OrderedDict
from numba import njit

from cued.utility import FrequencyContainers, TimeContainers, ScreeningContainers, ParamsParser
from cued.utility import ConversionFactors as CoFa
from cued.utility import MpiHelpers, rmdir_mkdir_chdir, chdir
from cued.plotting import write_and_compile_latex_PDF, read_dataset
from cued.kpoint_mesh import hex_mesh, rect_mesh
from cued.observables import *
from cued.plotting import write_and_compile_screening_latex_PDF
from cued.rhs_ode import *

import sys as system

def sbe_solver(sys, params):
	"""
	Function that initializes MPI-parallelization and distributes parameters that are given as a
	list in the params.py file to the individual MPI-processes. Runs the SBE-calculation for each
	parameter combination.

	Parameters
	----------
	sys : class
		Symbolic Hamiltonian of the system
	params : class
		parameters of the params.py file
	"""

	# Initialize Mpi and parse params
	P = ParamsParser(params)
	Mpi = MpiHelpers()

	P.n = sys.n
	P.n_sheets = 1
	if hasattr(sys, 'n_sheets'):
		P.n_sheets = sys.n_sheets

	P.combined_parallelization = False

	# Parallelize over paths and parameters if the following conditions are met:
	#  - Number of tasks % Nk2 = 0
	#  - Number of params % (number of tasks / Nk2 ) == 0
	#  - Number of tasks <= Nk2 * number of params
	if Mpi.size > params.Nk2 and Mpi.size > P.number_of_combinations and not P.parallelize_over_points:
		if Mpi.size % params.Nk2 == 0 and P.number_of_combinations % (Mpi.size/params.Nk2) == 0 and Mpi.size <= P.number_of_combinations*params.Nk2:
			if Mpi.rank == 0:
				print("Parallelization over paths and parameters\n Warning: You need to know what you are doing.")
				print("In case you choose "+str(P.number_of_combinations)+" MPI ranks, the parallelization will be much more straightforward.")
			Mpi.params_sets = int(Mpi.size/params.Nk2)
			Mpi.color = Mpi.rank//params.Nk2
			Mpi.mod = Mpi.rank%params.Nk2
			Mpi.local_params_idx_list = list(range(P.number_of_combinations))[Mpi.color::Mpi.params_sets]
			P.combined_parallelization = True
			P.path_parallelization = True
		else:
			ncpu_list = []
			for i in range(2, 1+P.number_of_combinations):
				if P.number_of_combinations % (i) == 0:
					ncpu_list.append(i*params.Nk2)
			if Mpi.rank == 0:
				print("For parallelization over paths and parameters, choose "
						+ "the number of MPI ranks from " + str(ncpu_list) )
			system.exit()

	# Parallelize over parameters if there are more parameter combinations than paths
	elif P.number_of_combinations >= params.Nk2 or P.path_list and not P.parallelize_over_points:
		Mpi.mod = None
		Mpi.local_params_idx_list = Mpi.get_local_idx(P.number_of_combinations)
		P.path_parallelization = False

	# Parallelize over paths else
	else:
		Mpi.mod = None
		Mpi.local_params_idx_list = range(P.number_of_combinations)
		P.path_parallelization = True

	# make subcommunicator
	P.Nk1 = params.Nk1
	P.Nk2 = params.Nk2
	make_subcommunicators(Mpi, P)

	#run sbe for i'th parameter set
	for i in Mpi.local_params_idx_list:
		P.distribute_parameters(i, params)
		run_sbe(sys, P, Mpi)

	# Wait until all calculations are finished.
	if P.save_screening or P.save_latex_pdf:
		write_screening_combinations_mpi(P, params, Mpi)

def make_subcommunicators(Mpi, P):

	if P.combined_parallelization:

		Mpi.subcomm = Mpi.comm.Split(Mpi.color, Mpi.rank)
		Mpi.local_Nk2_idx_list = [Mpi.mod]

	elif P.parallelize_over_points:	#works only for fixed Nk1 and Nk2
		Mpi.local_Nk2_idx_list = Mpi.get_local_idx(P.Nk2*P.Nk1)
		Mpi.subcomm = Mpi.comm.Split(0, Mpi.rank)

	elif P.path_parallelization:
		Mpi.local_Nk2_idx_list = Mpi.get_local_idx(P.Nk2)
		Mpi.subcomm = Mpi.comm.Split(0, Mpi.rank)

	else:
		Mpi.local_Nk2_idx_list = np.arange(P.Nk2)
		Mpi.subcomm = Mpi.comm.Split(Mpi.rank, Mpi.rank)


def run_sbe(sys, P, Mpi):
	"""
	line numberSolver for the semiconductor bloch equation ( eq. (39) or (47) in https://arxiv.org/abs/2008.03177)
	for a n band system with numerical calculation of the dipole elements (analytical dipoles
	can be used for n=2)

	Parameters
	----------
	sys : class
		Symbolic Hamiltonian of the system
	P : class
		Default parameters combined with user parameters from the params.py file
	Mpi : class
		Information needed for MPI-parallel runs

	Returns
	-------
	params.txt
		.txt file containing the parameters of the calculation

	time_data.dat
		.dat file containing the time-dependent observables

	frequency_data.dat
		.dat file containing the frequency-dependent observables

	"""
	# Start time of sbe_solver
	start_time = time.perf_counter()

	# Make Brillouin zone (saved in P)
	make_BZ(P)

	# USER OUTPUT
	###########################################################################
	if P.user_out:
		print_user_info(P)

	# INITIALIZATIONS
	###########################################################################

	# Make containers for time- and frequency- dependent observables
	T = TimeContainers(P)
	W = FrequencyContainers()

	# Make rhs of ode for 2band or nband solver; returns 0 for series expansion
	sys.eigensystem_dipole_path(P.paths[0], P) # change structure, such that hfjit gets calculated first
	rhs_ode, solver = make_rhs_ode(P, T, sys)

	###########################################################################
	# SOLVING
	###########################################################################
	# Iterate through each path in the Brillouin zone
	for Nk2_idx in Mpi.local_Nk2_idx_list:
		path = P.paths[Nk2_idx]

		if P.user_out:
			print('Solving SBE for Path', Nk2_idx+1)

		# Evaluate the dipole components along the path
		sys.eigensystem_dipole_path(path, P)

		# Prepare calculations of observables
		current_exact_path, polarization_inter_path, current_intra_path =\
			prepare_current_calculations(path, Nk2_idx, P, sys)

		# Initialize the values of of each k point vector

		y0 = initial_condition(P, sys.e_in_path)
		y0 = np.append(y0, [0.0])

		# Set the initual values and function parameters for the current kpath
		if P.dm_dynamics_method in ('sbe', 'semiclassics'):
			if P.solver_method in ('bdf', 'adams'):
				solver.set_initial_value(y0, P.t0)\
					.set_f_params(path, sys.dipole_in_path, sys.e_in_path, y0, P.dk)
			elif P.solver_method == 'rk4':
				T.solution_y_vec[:] = y0
		elif P.dm_dynamics_method in ('series_expansion', 'EEA'):
			T.solution_y_vec = np.copy(y0)
			T.time_integral = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_complex_np)
		# Propagate through time
		# Index of current integration time step
		ti = 0
		solver_successful = True

		while solver_successful and ti < P.Nt:
			# User output of integration progress
			if (ti % (P.Nt//20) == 0 and P.user_out):
				print('{:5.2f}%'.format((ti/P.Nt)*100))

			calculate_solution_at_timestep(solver, Nk2_idx, ti, T, P, Mpi)

			# Calculate the currents at the timestep ti
			calculate_currents(ti, current_exact_path, polarization_inter_path, current_intra_path, T, P)

			# Integrate one integration time step
			if P.dm_dynamics_method in ('sbe', 'semiclassics'):
				if P.solver_method in ('bdf', 'adams'):
					solver.integrate(solver.t + P.dt)
					solver_successful = solver.successful()

				elif P.solver_method == 'rk4':
					T.solution_y_vec = rk_integrate(T.t[ti], T.solution_y_vec, path, sys,
													y0, P.dk, P.dt, rhs_ode)

			elif P.dm_dynamics_method in ('series_expansion', 'EEA'):
				T.solution_y_vec[:-1], T.time_integral = von_neumann_series(T.t[ti], T.A_field[ti], T.E_field[ti], path, sys, y0[:-1], T.time_integral, P, ti)

			# Increment time counter
			ti += 1

	# in case of MPI-parallel execution: mpi sum
	mpi_sum_currents(T, P, Mpi)

	# End time of solver loop
	end_time = time.perf_counter()
	P.run_time = end_time - start_time

	# calculate and write solutions
	update_currents_with_kweight(T, P)
	calculate_fourier(T, P, W)
	write_current_emission_mpi(T, P, W, sys, Mpi)

	# Save the parameters of the calculation
	params_name = P.header + 'params.txt'
	paramsfile = open(params_name, 'w')
	paramsfile.write("Runtime: {:.16f} s \n\n".format(P.run_time))
	exceptions = {'__weakref__', '__doc__', '__dict__', '__module__', \
				'_ParamsParser__user_defined_field', 'header', 'mesh', 'number_of_combinations', \
				'params_combinations', 'params_lists', 'path_list', 'paths', 'run_time', \
				't_pdf_densmat', 'tail', 'type_complex_np', 'type_real_np', 'user_params'}
	for key in sorted(P.__dict__.keys() - exceptions):
		paramsfile.write(str(key) + ' = ' + str(P.__dict__[key]) + "\n")

	paramsfile.close()

	if P.save_full:
		# write_full_density_mpi(T, P, sys, Mpi)
		S_name = 'Sol_' + P.tail
		np.savez(S_name, t=T.t, solution_full=T.solution_full, paths=P.paths,
				 electric_field=T.electric_field(T.t), A_field=T.A_field)

	#save density matrix at given points in time
	if P.save_dm_t:
		np.savez(P.header + 'time_matrix', pdf_densmat=T.pdf_densmat,
			 t_pdf_densmat=T.t_pdf_densmat, A_field=T.A_field)


def make_BZ(P):
		# Form Brillouin Zone
	if P.BZ_type == 'hexagon':
		if P.align == 'K':
			P.E_dir = np.array([1, 0], P.type_real_np)
		elif P.align == 'M':
			P.E_dir = np.array([np.cos(np.radians(-30)),
									np.sin(np.radians(-30))], dtype=P.type_real_np)
		P.dk, P.kweight, P.paths, P.mesh = hex_mesh(P)

	elif P.BZ_type == 'rectangle':
		P.E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
								np.sin(np.radians(P.angle_inc_E_field))], dtype=P.type_real_np)
		P.dk, P.kweight, P.paths, P.mesh = rect_mesh(P)

	P.E_ort = np.array([P.E_dir[1], -P.E_dir[0]])

	if P.parallelize_over_points:
		if P.gauge != 'velocity':
			system.exit('Parallelization over points can only be used with the velocity gauge')
		Nk1_buf = np.copy(P.Nk1)
		Nk2_buf = np.copy(P.Nk2)
		paths_buf = np.copy(P.paths)

		P.paths = np.empty((Nk1_buf*Nk2_buf, 1, 2))
		for i in range(Nk2_buf):
			for j in range(Nk1_buf):
				P.paths[Nk1_buf*i + j, 0, 0] = paths_buf[i, j, 0]
				P.paths[Nk1_buf*i + j, 0, 1] = paths_buf[i, j, 1]
		P.Nk1 = 1
		P.Nk2 = Nk1_buf * Nk2_buf

def make_rhs_ode(P, T, sys):

	if P.dm_dynamics_method in ('sbe', 'semiclassics'):
		if P.solver == '2band':
			if P.n != 2:
				raise AttributeError('2-band solver works for 2-band systems only')
			else:
				rhs_ode = make_rhs_ode_2_band(sys, T.electric_field, P)

		elif P.solver == 'nband':
			rhs_ode = make_rhs_ode_n_band(sys, T.electric_field, P)
		else:
			rhs_ode = 0

		if P.solver_method in ('bdf', 'adams'):
			solver = ode(rhs_ode, jac=None)\
				.set_integrator('zvode', method=P.solver_method, max_step=P.dt)
		else:
			solver = 0

	else:
		rhs_ode = 0
		solver = 0

	return rhs_ode, solver


def prepare_current_calculations(path, Nk2_idx, P, sys):

	polarization_inter_path = None
	current_intra_path = None
	if sys.system == 'ana':
		if P.gauge == 'length':
			current_exact_path = make_emission_exact_path_length(path, P, sys)
		if P.gauge == 'velocity':
			current_exact_path = make_emission_exact_path_velocity(path, P, sys)
		if P.split_current:
			polarization_inter_path = make_polarization_path(path, P, sys)
			current_intra_path = make_current_path(path, P, sys)
	elif sys.system == 'num':
		if P.gauge == 'length':
			current_exact_path = make_current_exact_path_hderiv_length(path, P, sys)
		if P.gauge == 'velocity':
			current_exact_path = make_current_exact_path_hderiv_velocity(path, P, sys)
		if P.split_current:
			if P.gauge == 'length':
				polarization_inter_path = make_polarization_inter_path_length(P, sys)
				current_intra_path = make_intraband_current_path_length(path, P, sys)
			if P.gauge == 'velocity':
				polarization_inter_path = make_polarization_inter_path_velocity(path, P, sys)
				current_intra_path = make_intraband_current_path_velocity(path, P, sys)
	else:
		current_exact_path = make_current_exact_bandstructure(path, P, sys)
		if P.split_current:
			polarization_inter_path = make_polarization_inter_bandstructure(P, sys)
			current_intra_path = make_intraband_current_bandstructure(path, P, sys)
	return current_exact_path, polarization_inter_path, current_intra_path


def calculate_solution_at_timestep(solver, Nk2_idx, ti, T, P, Mpi):

	is_first_Nk2_idx = (Mpi.local_Nk2_idx_list[0] == Nk2_idx)

	if P.dm_dynamics_method in ('sbe', 'semiclassics'):
		if P.solver_method in ('bdf', 'adams'):
			# Do not append the last element (A_field)
			T.solution = solver.y[:-1].reshape(P.Nk1, P.n, P.n)

			# Construct time array only once
			if is_first_Nk2_idx:
				# Construct time and A_field only in first round
				T.t[ti] = solver.t
				T.A_field[ti] = solver.y[-1].real
				T.E_field[ti] = T.electric_field(T.t[ti])

		elif P.solver_method == 'rk4':
			# Do not append the last element (A_field)
			T.solution = T.solution_y_vec[:-1].reshape(P.Nk1, P.n, P.n)

			# Construct time array only once
			if is_first_Nk2_idx:
				# Construct time and A_field only in first round
				T.t[ti] = ti*P.dt + P.t0
				T.A_field[ti] = T.solution_y_vec[-1].real
				T.E_field[ti] = T.electric_field(T.t[ti])

	elif P.dm_dynamics_method in ('series_expansion', 'EEA'):

		# Do not append the last element (A_field)
		T.solution = T.solution_y_vec[:-1].reshape(P.Nk1, P.n, P.n)

		# Construct time array only once
		if is_first_Nk2_idx:
			# Construct time and A_field only in first round
			T.t[ti] = ti*P.dt + P.t0
			T.E_field[ti] = T.electric_field(T.t[ti])
			T.A_field[ti] = T.A_field[ti-1] - T.electric_field(T.t[ti])*P.dt

	# Only write full density matrix solution if save_full is True
	if P.save_full:
		T.solution_full[:, Nk2_idx, ti, :, :] = T.solution

	# store density matrix for Latex pdf
	if P.save_latex_pdf or P.save_dm_t:
		store_density_matrix_for_pdf(T, P, Nk2_idx, ti)


def store_density_matrix_for_pdf(T, P, Nk2_idx, ti):

	for count, t_pdf_densmat in enumerate(P.t_pdf_densmat):
		if (t_pdf_densmat > T.t[ti-1] and t_pdf_densmat < T.t[ti]) or t_pdf_densmat == T.t[ti]:
			T.pdf_densmat[:, Nk2_idx, count, :, :] = T.solution
			T.t_pdf_densmat[count] = T.t[ti]


def calculate_currents(ti, current_exact_path, polarization_inter_path, current_intra_path, T, P):

	j_E_dir_buf, j_ortho_buf = current_exact_path(T.solution, T.E_field[ti], T.A_field[ti])

	T.j_E_dir[ti] += j_E_dir_buf
	T.j_ortho[ti] += j_ortho_buf

	if P.split_current:
		P_E_dir_buf, P_ortho_buf = polarization_inter_path(T.solution, T.E_field[ti], T.A_field[ti])
		j_intra_E_dir_buf, j_intra_ortho_buf, j_anom_ortho_buf = current_intra_path(T.solution, T.E_field[ti], T.A_field[ti])

		T.P_E_dir[ti] += P_E_dir_buf
		T.P_ortho[ti] += P_ortho_buf
		T.j_intra_E_dir[ti] += j_intra_E_dir_buf
		T.j_intra_ortho[ti] += j_intra_ortho_buf
		T.j_anom_ortho[ti, :] += j_anom_ortho_buf

def rk_integrate(t, y, kpath, sys, y0, dk, dt, rhs_ode):

	k1 = rhs_ode(t,          y,          kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
	k2 = rhs_ode(t + 0.5*dt, y + 0.5*k1, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
	k3 = rhs_ode(t + 0.5*dt, y + 0.5*k2, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
	k4 = rhs_ode(t +     dt, y +     k3, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)

	ynew = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

	return ynew

@njit
def y0deriv(y, dk, Nk_path, n, dk_order, type_complex_np):

	diffy0 = np.zeros((Nk_path, n, n), dtype=type_complex_np)
	for i in range(n):
		for k in range(Nk_path):
			right4 = (k+4)
			right3 = (k+3)
			right2 = (k+2)
			right  = (k+1)
			left   = (k-1)
			left2  = (k-2)
			left3  = (k-3)
			left4  = (k-4)
			if k == 0:
				left   = (Nk_path-1)
				left2  = (Nk_path-2)
				left3  = (Nk_path-3)
				left4  = (Nk_path-4)
			elif k == 1 and dk_order >= 4:
				left2  = (Nk_path-1)
				left3  = (Nk_path-2)
				left4  = (Nk_path-3)
			elif k == 2 and dk_order >= 6:
				left3  = (Nk_path-1)
				left4  = (Nk_path-2)
			elif k == 3 and dk_order >= 8:
				left4  = (Nk_path-1)
			elif k == Nk_path-1:
				right4 = 3
				right3 = 2
				right2 = 1
				right  = 0
			elif k == Nk_path-2 and dk_order >= 4:
				right4 = 2
				right3 = 1
				right2 = 0
			elif k == Nk_path-3 and dk_order >= 6:
				right4 = 1
				right3 = 0
			elif k == Nk_path-4 and dk_order >= 8:
				right4 = 0

			if dk_order == 2:
				diffy0[k, i, i]   += ( y[right, i, i]/2   - y[left, i, i]/2  ) / dk
			elif dk_order == 4:
				diffy0[k, i, i]   += (- y[right2, i, i]/12   + 2/3*y[right, i, i]   - 2/3*y[left, i, i]   + y[left2, i, i]/12 ) / dk
			elif dk_order == 6:
				diffy0[k, i, i]   += (  y[right3, i, i]/60   - 3/20*y[right2, i, i]   + 3/4*y[right, i, i] \
					- y[left3, i, i]/60    + 3/20*y[left2, i, i]    - 3/4*y[left, i, i] ) / dk
			elif dk_order == 8:
				diffy0[k, i, i]   += (- y[right4, i, i]/280   + 4/105*y[right3, i, i]   - 1/5*y[right2, i, i]   + 4/5*y[right, i, i] \
					+ y[left4, i, i] /280   - 4/105*y[left3, i, i]    + 1/5*y[left2, i, i]    - 4/5*y[left, i, i] ) / dk


	return diffy0

def von_neumann_series(t, A_field, E_field, path, sys, y0, time_integral, P, ti):

	# rescale solution vector and initial condition to be a matrix
	y_mat = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_complex_np)
	y0_mat = y0.reshape(P.Nk1, P.n, P.n)

	# 0th order
	y_mat[:, :, :] = np.copy(y0_mat[:, :, :])

	if P.dm_dynamics_method == 'EEA':       # Approximate formula for DC remnants (Int(E^2*A), needs velocity gauge to make sense for finite times!)

		if P.first_order:
			y_mat = first_order_taylor(y_mat, y0_mat, t, E_field, A_field, sys.dipole_in_path, sys.e_in_path, \
							sys.dipole_derivative_in_path, sys.bandstructure_derivative_in_path, P.T2, P.n)
		if P.second_order:
			y_mat, time_integral = second_order_taylor(y_mat, time_integral, y0_mat, E_field, A_field, \
							sys.dipole_in_path, sys.dipole_derivative_in_path, P.T2, P.dt, P.n, P.Nk1)

	elif P.dm_dynamics_method == 'series_expansion':
		# calculate eigenvalues and dipole elements at current time step (velocity gauge!)
		path_after_shift = np.copy(path)

		if not P.gauge == 'length' or not P.linear_response:
			path_after_shift[:, 0] = path[:, 0] + A_field*P.E_dir[0]
			path_after_shift[:, 1] = path[:, 1] + A_field*P.E_dir[1]

			sys.eigensystem_dipole_path(path_after_shift, P)

		if P.gauge == 'length' :
			if ti == 0:
				P.diffy0 = y0deriv(y0_mat, P.dk, P.Nk1, P.n, P.dk_order, P.type_complex_np)

		if P.first_order:
			if P.high_damping:
				y_mat = first_order_high_damping(y_mat, y0_mat, t, E_field, sys.e_in_path, sys.dipole_in_path, P.T2, P.n)
			else:
				y_mat, time_integral = first_order(y_mat, time_integral, y0_mat, t, E_field, A_field, sys.e_in_path, sys.dipole_in_path, P.T2, P.dt, P.n, P.gauge, P.diffy0)
		if P.second_order:
			if P.high_damping:
				y_mat, time_integral = second_order(y_mat, time_integral, y0_mat, E_field, sys.dipole_in_path, P.T2, P.dt, P.n, P.Nk1)
			else:
				print('Warning: second order without high damping not implemented yet!')
	return y_mat.flatten('C'), time_integral


@njit
def first_order(y_mat, time_integral, y0_mat, t, E_field, A_field, e_in_path, dipole_in_path, T2, dt, n, gauge, diffy0):

	if gauge == 'length':
		for i in range(n):
			y_mat[:, i, i] -= diffy0[:, i, i] * A_field

	for i in range(n):
		for j in range(n):
			time_integral[:, i, j] += np.exp( t * ( 1j * ( e_in_path[:, i] - e_in_path[:, j] ) +  1/T2  ) ) * E_field * dipole_in_path[:, i, j] * dt
			if i != j:
				y_mat[:, i, j] -= 1j * time_integral[:, i, j] * (y0_mat[:, i, i] - y0_mat[:, j, j]) * np.exp( - t * ( 1j * ( e_in_path[:, i] - e_in_path[:, j] ) + 1/T2  ) )

	return y_mat, time_integral


@njit
def first_order_high_damping(y_mat, y0_mat, t, E_field, e_in_path, dipole_in_path, T2, n):
	for i in range(n):
		for j in range(n):
			if i!= j:
				y_mat[:, i, j] -= 1j * T2 * ( y0_mat[:, i, i] - y0_mat[:, j, j] ) \
					* E_field * dipole_in_path[:, i, j] 

	return y_mat

@njit
def second_order_high_damping(y_mat, time_integral, y0_mat, E_field, dipole_in_path, T2, dt, n, Nk1):

	for i in range(n):
		for k in range(n):
			for nk in range(Nk1):
				time_integral[nk, i, k] += np.abs(E_field * dipole_in_path[nk, k, i])**2 * dt
				y_mat[nk, i, i] -= T2 * ( y0_mat[nk, i, i] - y0_mat[nk, k, k] ) * time_integral[nk, i, k]

	return y_mat, time_integral

@njit
def first_order_taylor(y_mat, y0_mat, t, E_field, A_field, dipole_in_path, e_in_path, dipole_derivative_in_path, bandstructure_derivative_in_path, T2, n):

	for i in range(n):
		for j in range(n):
			# first order of taylor expansion
			y_mat[:, i, j] -= 1j / (1/T2 + 1j*( e_in_path[:, i] - e_in_path[:, j] )) * ( y0_mat[:, i, i] - y0_mat[:, j, j] ) \
				* E_field * dipole_in_path[:, i, j] #* np.exp(1j * t * ( e_in_path[:, i] - e_in_path[:, j] ) ) \
				

			# first order of taylor expansion
			#y_mat[:, i, j] += dipole_derivative_in_path[:, k, i] * E_field * A_field \
			#	* 1j * t * E_field * A_field * dipole_in_path[:, i, j] \
			#	* ( bandstructure_derivative_in_path[:, i] - bandstructure_derivative_in_path[:, i] )

	return y_mat

@njit
def second_order_taylor(y_mat, time_integral, y0_mat, E_field, A_field, dipole_in_path, dipole_derivative_in_path, T2, dt, n, Nk1):

	for i in range(n):
		for k in range(n):
			for nk in range(Nk1):
				# first order of taylor expansion
				# time_integral[nk, i, k] += np.abs(E_field * dipole_in_path[nk, i, k])**2

				# second order of taylor expansion
				C = 2 * np.real(dipole_in_path[nk, k, i] * dipole_derivative_in_path[nk, k, i])
				time_integral[nk, i, k] += C * E_field**2 * A_field * dt
				y_mat[nk, i, i] -= T2 * ( y0_mat[nk, i, i] - y0_mat[nk, k, k] ) * time_integral[nk, i, k]

	return y_mat, time_integral

def initial_condition(P, e_in_path):
	'''
	Occupy conduction band according to inital Fermi energy and temperature
	'''
	num_kpoints = e_in_path[:, 0].size
	num_bands = e_in_path[0, :].size
	distrib_bands = np.zeros([num_kpoints, num_bands], dtype=P.type_complex_np)
	initial_condition = np.zeros([num_kpoints, num_bands, num_bands], dtype=P.type_complex_np)
	if P.temperature > 1e-5:
		distrib_bands += 1/(np.exp((e_in_path-P.e_fermi)/P.temperature) + 1)
	else:
		smaller_e_fermi = (P.e_fermi - e_in_path) > 0
		distrib_bands[smaller_e_fermi] += 1

	for k in range(num_kpoints):
		initial_condition[k, :, :] = np.diag(distrib_bands[k, :])
	return initial_condition.flatten('C')


def diff(x, y):
	'''
	Takes the derivative of y w.r.t. x
	'''
	if len(x) != len(y):
		raise ValueError('Vectors have different lengths')
	if len(y) == 1:
		return 0

	dx = np.roll(x,-1) - np.roll(x,1)
	dy = np.roll(y,-1) - np.roll(y,1)

	return dy/dx


def fourier(dt, data):
	'''
	Calculate the phase correct fourier transform with proper normalization
	for calculations centered around t=0
	'''
	return (dt/np.sqrt(2*np.pi))*fftshift(fft(ifftshift(data)))


def ifourier(dt, data):
	'''
	Calculate the phase correct inverse fourier transform with proper normalization
	for calculations centered around t=0
	'''
	return (np.sqrt(2*np.pi)/dt)*fftshift(ifft(ifftshift(data)))


def gaussian(t, sigma, tau):
	'''
	Window function to multiply a Function f(t) before Fourier transform
	to ensure no step in time between t_final and t_final + delta
	'''
	return np.exp(-(t-tau)**2/sigma**2)

def hann(t):
	'''
	Window function to multiply a Function f(t) before Fourier transform
	to ensure no step in time between t_final and t_final + delta
	'''
	return (np.cos(np.pi*t/(np.amax(t)-np.amin(t))))**2

def parzen(t):
	'''
	Window function to multiply a Function f(t) before Fourier transform
	to ensure no step in time between t_final and t_final + delta
	'''
	n_t = t.size
	t_half_size = (t[-1]-t[0])/2
	t_1 = t[0     :n_t//4]
	t_2 = t[n_t//4:n_t//2]

	parzen                = np.zeros(n_t)
	parzen[0:     n_t//4] = 2*(1-np.abs(t_1)/t_half_size)**3
	parzen[n_t//4:n_t//2] = 1-6*(t_2/t_half_size)**2*(1-np.abs(t_2)/t_half_size)
	parzen                = parzen + parzen[::-1]
	parzen[n_t//2]        = 1.0

	return parzen

def mpi_sum_currents(T, P, Mpi):

	T.j_E_dir       = Mpi.sync_and_sum(T.j_E_dir)
	T.j_ortho       = Mpi.sync_and_sum(T.j_ortho)
	if P.split_current:
		T.j_intra_E_dir = Mpi.sync_and_sum(T.j_intra_E_dir)
		T.j_intra_ortho = Mpi.sync_and_sum(T.j_intra_ortho)
		T.P_E_dir       = Mpi.sync_and_sum(T.P_E_dir)
		T.P_ortho       = Mpi.sync_and_sum(T.P_ortho)
		T.j_anom_ortho  = Mpi.sync_and_sum(T.j_anom_ortho)
	if P.save_latex_pdf:
		T.pdf_densmat   = Mpi.sync_and_sum(T.pdf_densmat)

def update_currents_with_kweight(T, P):

	T.j_E_dir *= P.kweight
	T.j_ortho *= P.kweight

	if P.split_current:
		T.j_intra_E_dir *= P.kweight
		T.j_intra_ortho *= P.kweight

		T.dtP_E_dir = diff(T.t, T.P_E_dir)*P.kweight
		T.dtP_ortho = diff(T.t, T.P_ortho)*P.kweight

		T.P_E_dir *= P.kweight
		T.P_ortho *= P.kweight

		T.j_anom_ortho *= P.kweight

		# Eq. (81) SBE formalism paper
		T.j_deph_E_dir = 1/P.T2*T.P_E_dir
		T.j_deph_ortho = 1/P.T2*T.P_ortho

		T.j_intra_plus_dtP_E_dir = T.j_intra_E_dir + T.dtP_E_dir
		T.j_intra_plus_dtP_ortho = T.j_intra_ortho + T.dtP_ortho

		T.j_intra_plus_anom_ortho = T.j_intra_ortho
		for i in range(P.n):
			T.j_anom_ortho_full += T.j_anom_ortho[:, i]
			T.j_intra_plus_anom_ortho += T.j_anom_ortho[:, i]

	if P.sheet_current:
		for i in range(P.n_sheets):
			for j in range(P.n_sheets):
				T.j_E_dir_full += T.j_E_dir[:, i, j]
				T.j_ortho_full += T.j_ortho[:, i, j]

def calculate_fourier(T, P, W):

	# Fourier transforms
	# 1/(3c^3) in atomic units
	prefac_emission = 1/(3*(137.036**3))
	dt_out = T.t[1] - T.t[0]
	ndt_fft = (T.t.size-1)*P.factor_freq_resolution + 1
	W.freq = fftshift(fftfreq(ndt_fft, d=dt_out))

	if P.fourier_window_function == 'gaussian':
		T.window_function = gaussian(T.t, P.gaussian_window_width,P.gaussian_center)
	elif P.fourier_window_function == 'hann':
		T.window_function = hann(T.t)
	elif P.fourier_window_function == 'parzen':
		T.window_function = parzen(T.t)

	W.I_E_dir, W.j_E_dir =\
		fourier_current_intensity(T.j_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
	W.I_ortho, W.j_ortho =\
		fourier_current_intensity(T.j_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

	# always compute the Fourier transform with hann and parzen window for comparison; this is printed to the latex PDF
	W.I_E_dir_hann, W.j_E_dir_hann =\
		fourier_current_intensity(T.j_E_dir, hann(T.t), dt_out, prefac_emission, W.freq, P)
	W.I_ortho_hann, W.j_ortho_hann =\
		fourier_current_intensity(T.j_ortho, hann(T.t), dt_out, prefac_emission, W.freq, P)

	W.I_E_dir_parzen, W.j_E_dir_parzen =\
		fourier_current_intensity(T.j_E_dir, parzen(T.t), dt_out, prefac_emission, W.freq, P)
	W.I_ortho_parzen, W.j_ortho_parzen =\
		fourier_current_intensity(T.j_ortho, parzen(T.t), dt_out, prefac_emission, W.freq, P)

	if P.split_current:
		# Approximate current and emission intensity
		W.I_intra_plus_dtP_E_dir, W.j_intra_plus_dtP_E_dir =\
			fourier_current_intensity(T.j_intra_plus_dtP_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_intra_plus_dtP_ortho, W.j_intra_plus_dtP_ortho =\
			fourier_current_intensity(T.j_intra_plus_dtP_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

		# Intraband current and emission intensity
		W.I_intra_E_dir, W.j_intra_E_dir =\
			fourier_current_intensity(T.j_intra_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_intra_ortho, W.j_intra_ortho =\
			fourier_current_intensity(T.j_intra_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

		# Polarization-related current and emission intensity
		W.I_dtP_E_dir, W.dtP_E_dir =\
			fourier_current_intensity(T.dtP_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_dtP_ortho, W.dtP_ortho =\
			fourier_current_intensity(T.dtP_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

		# Dephasing current and emission intensity
		W.I_deph_E_dir, W.j_deph_E_dir =\
			fourier_current_intensity(T.j_deph_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_deph_ortho, W.j_deph_ortho =\
			fourier_current_intensity(T.j_deph_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

		# Anomalous current, intraband current (de/dk-related) + anomalous current; and emission int.
		W.I_anom_ortho, W.j_anom_ortho =\
			fourier_current_intensity(T.j_anom_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_anom_ortho_full, W.j_anom_ortho_full =\
			fourier_current_intensity(T.j_anom_ortho_full, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_intra_plus_anom_ortho, W.j_intra_plus_anom_ortho =\
			fourier_current_intensity(T.j_intra_plus_anom_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

	if P.sheet_current:
		W.I_E_dir_full, W.j_E_dir_full =\
			fourier_current_intensity(T.j_E_dir_full, T.window_function, dt_out, prefac_emission, W.freq, P)
		W.I_ortho_full, W.j_ortho_full =\
			fourier_current_intensity(T.j_ortho_full, T.window_function, dt_out, prefac_emission, W.freq, P)

	if P.gabor_transformation:
		if not(hasattr(P,"gabor_gaussian_center") and hasattr(P,"gabor_window_width")):
			system.exit("Either no center(s) or width(s) were given for the invoked Gabor transformation.")
		W.I_E_dir_GT = []
		W.j_E_dir_GT = []
		W.I_ortho_GT = []
		W.j_ortho_GT = []

		for center in P.gabor_gaussian_center:
			W.I_E_dir_CT = []
			W.j_E_dir_CT = []
			W.I_ortho_CT = []
			W.j_ortho_CT = []

			for window in P.gabor_window_width:
				T.window_function = gaussian(T.t, window,center)
				I_E_dir, j_E_dir=\
					fourier_current_intensity(T.j_E_dir, T.window_function, dt_out, prefac_emission, W.freq, P)
				I_ortho, j_ortho =\
					fourier_current_intensity(T.j_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)
				W.I_E_dir_CT.append(I_E_dir)
				W.j_E_dir_CT.append(j_E_dir)
				W.I_ortho_CT.append(I_ortho)
				W.j_ortho_CT.append(j_ortho)

			W.I_E_dir_GT.append(W.I_E_dir_CT)
			W.j_E_dir_GT.append(W.j_E_dir_CT)
			W.I_ortho_GT.append(W.I_ortho_CT)
			W.j_ortho_GT.append(W.j_ortho_CT)
# def write_full_density_mpi(T, P, sys, Mpi):
#     relative_dir = 'densities'
#     if Mpi.rank == 0:
#         os.mkdir(relative_dir)
#     Mpi.comm.Barrier()
#     # Write out density matrix for every path
#     for Nk2_idx in Mpi.local_Nk2_idx_list:
#         write_full_density(T, P, sys, Nk2_idx, relative_dir)

# def write_full_density(T, P, sys, Nk2_idx, relative_dir):

#     path = P.paths[Nk2_idx]
#     kpath_filename = relative_dir + 'kpath_path_idx_{:d}'
#     full_density_filename = relative_dir + 'density_data_path_idx_{:d}.dat'.format(Nk2_idx)
#     # Upper triangular, coherence entries
#     nt = int((P.n/2)*(P.n-1))
#     # Diagonal, density entries
#     nd = P.n
#     dens_header = ("{:25s} {:27s} {:27s}" + " {:27s}"*nd)
#     dens_header_format = ["rho({:d},{:d})".format(idx, idx) for idx in range(nd)]
#     dens_header = dens_header.format("t", "kx", "ky", *dens_header_format)

#     cohe_header = (" {:27s} {:27s}"*nt)
#     cohe_header_format = []
#     for idx in range(nd):
#         for jdx in range(idx + 1, nd):
#             cohe_header_format.append("Re[rho({:d},{:d})]".format(idx, jdx))
#             cohe_header_format.append("Im[rho({:d},{:d})]".format(idx, jdx))

#     cohe_header = cohe_header.format(*cohe_header_format)
#     full_density_header = dens_header + cohe_header
#     full_density

def write_screening_combinations_mpi(P, params, Mpi):
	# Wait until all jobs are finished
	Mpi.comm.Barrier()
	if Mpi.rank == 0 and P.number_of_combinations > 1:
		write_screening_combinations(P, params)

def write_screening_combinations(P, params):
	'''
	Write screening data from output files generated in the main code.
	'''
	# Check which parameters are given as lists or ndarrays
	# Needs to be Ordered! (see __combine_parameters in params_parser.py)
	# parameter values is an empty Ordered Dictionary
	# needed for construction of the screening file name
	screening_filename_template = ''
	params_dims = ()
	params_values = OrderedDict()
	for key, item in OrderedDict(params.__dict__).items():
		if type(item) == list or type(item) == np.ndarray:
			params_values[key] = list(item)
			params_dims += (len(item), )
			screening_filename_template += key + '={' + key + '}' + '_'

	# Create all matrix indices
	params_idx = [np.unravel_index(i, params_dims) for i in range(P.number_of_combinations)]

	# Load a reference f/f0 into memory
	P.construct_current_parameters_and_header(0, params)
	_t, freq_data, _d = read_dataset(path='.', prefix=P.header)

	# First E-dir, second ortho, third combined data
	S = np.empty(3, dtype=ScreeningContainers)
	S[0] = ScreeningContainers(freq_data['f/f0'], params_dims, P.plot_format)
	S[1] = ScreeningContainers(freq_data['f/f0'], params_dims, P.plot_format)
	S[2] = ScreeningContainers(freq_data['f/f0'], params_dims, P.plot_format)

	# Load all f/f0 and intensities into memory
	for i, idx in enumerate(params_idx):
		P.construct_current_parameters_and_header(i, params)
		# example:
		# if E0 = [1, 2], chirp = [0, 1]
		# OrderedDict puts E0 before chirp
		# E0=1, chirp=0 -> (0, 0), E0=1, chirp=1 -> (0, 1)
		# E0=2, chirp=0 -> (1, 0), E0=2, chirp=1 -> (1, 1)
		_t, freq_data, _d = read_dataset(path='.', prefix=P.header)
		if not np.all(np.equal(S[0].ff0, freq_data['f/f0'])):
			raise ValueError("For screening plots, frequency scales of all parameters need to be equal.")
		S[0].full_screening_data[idx] = freq_data['I_E_dir']
		S[1].full_screening_data[idx] = freq_data['I_ortho']
		S[2].full_screening_data[idx] = freq_data['I_E_dir'] + freq_data['I_ortho']

	# Name elements of output file
	params_name = {}
	# the major parameter is the current y-axis of the screening-plot
	# we need to do this plot for every minor parameter (all others) combination
	for i, major_key in enumerate(params_values.keys()):
		# Generate index combinations of all minor parameters
		index_gen = [list(range(gen)) for j, gen in enumerate(params_dims) if j != i]
		# All indices except the major one
		idx_minor = np.delete(np.arange(len(params_dims)), i)

		# Index template to access the data array major parameter and data is [:]
		# if E0 = [1, 2], chirp = [0, 1] and we currently have E0 major
		# slice_template -> [:, 0, :] -> [:, 1, :]
		# meaning plot all E0 for chirp[0] -> all E0 for chirp[1]
		slice_template = np.empty(len(params_dims) + 1, dtype=object)
		slice_template[i] = slice(None)
		slice_template[-1] = slice(None)
		# In the file name we call the screening-parameter 'variable'
		params_name[major_key] = 'variable'
		for s in S:
			s.screening_parameter_name = major_key
			s.screening_parameter_values = params_values[major_key]

		# Header or column title in the .dat file
		screening_file_header_name = ['{}={}'.format(major_key, val) for val in params_values[major_key]]
		screening_file_header = ("{:25s}" + " {:27s}"*params_dims[i])\
			.format('f/f0', *screening_file_header_name)

		for idx_tuple in product(*index_gen):
			# idx_tuple only holds combinations of minor (non-screening param) indices
			# Now we create the output for every minor combination
			for j, idxm in enumerate(idx_minor):
				minor_key = list(params_values.keys())[idxm]
				params_name[minor_key] = list(params_values.values())[idxm][idx_tuple[j]]
				slice_template[idxm] = idx_tuple[j]
			for s in S:
				s.screening_output = s.full_screening_data[tuple(slice_template.tolist())]
			screening_foldername = screening_filename_template.format(**params_name)
			S[0].screening_filename = screening_foldername + 'E_dir_'
			S[1].screening_filename = screening_foldername + 'ortho_'
			S[2].screening_filename = screening_foldername + 'full_'
			rmdir_mkdir_chdir(screening_foldername + 'latex_pdf_files')
			if P.save_screening:
				for s in S:
					np.savetxt(s.screening_filename + 'intensity_freq_data.dat',
							   np.hstack((s.ff0[:, np.newaxis], s.screening_output.T)),
							   header=screening_file_header, delimiter=' '*3, fmt="%+.18e")
			if P.save_latex_pdf:
				write_and_compile_screening_latex_PDF(S)
			chdir()


def write_current_emission_mpi(T, P, W, sys, Mpi):

	# only save data from a single MPI rank
	if Mpi.rank == 0 or Mpi.mod == 0 or not P.path_parallelization:
		write_current_emission(T, P, W, sys, Mpi)

		if P.save_fields:
			write_efield_afield(T, P, W)


def write_current_emission(T, P, W, sys, Mpi):

	##################################################
	# Time data save
	##################################################
	if P.split_current:
		time_header = ("{:25s}" + " {:27s}"*10)\
			.format("t",
					"j_E_dir", "j_ortho",
					"j_intra_E_dir", "j_intra_ortho",
					"dtP_E_dir", "dtP_ortho", "j_deph_E_dir", "j_deph_ortho",
					"j_intra_plus_dtP_E_dir", "j_intra_plus_dtP_ortho")
		time_output = np.column_stack([T.t.real,
									   T.j_E_dir.real, T.j_ortho.real,
									   T.j_intra_E_dir.real, T.j_intra_ortho.real,
									   T.dtP_E_dir.real, T.dtP_ortho.real, T.j_deph_E_dir.real, T.j_deph_ortho.real,
									   T.j_intra_plus_dtP_E_dir.real, T.j_intra_plus_dtP_ortho.real])
		if P.save_anom:
			for i in range(P.n):
				time_header += (" {:27s}").format(f"j_anom_ortho[{i}]")
				time_output = np.column_stack((time_output, T.j_anom_ortho[:, i].real))
			time_header += (" {:27s}"*2).format("j_anom_ortho", "j_intra_plus_anom_ortho")
			time_output = np.column_stack((time_output, T.j_anom_ortho_full.real, T.j_intra_plus_anom_ortho.real))

	elif P.sheet_current:

		time_header =("{:25s}").format('t')
		time_output = T.t.real

		for i in range(P.n_sheets):
			for j in range(P.n_sheets):
				time_header += (" {:27s}"*2).format(f"j_E_dir[{i},{j}]", f"j_ortho[{i},{j}]")
				time_output = np.column_stack((time_output, T.j_E_dir[:, i, j].real, T.j_ortho[:, i, j].real) )
			time_header += (" {:27s}"*2).format("j_E_dir", "j_ortho")
			time_output = np.column_stack((time_output, T.j_E_dir_full.real, T.j_ortho_full.real) )
	else:
		time_header = ("{:25s}" + " {:27s}"*2)\
			.format("t", "j_E_dir", "j_ortho")
		time_output = np.column_stack([T.t.real,
									   T.j_E_dir.real, T.j_ortho.real])

	# Make the maximum exponent double digits
	time_output[np.abs(time_output) <= 10e-100] = 0
	time_output[np.abs(time_output) >= 1e+100] = np.inf

	np.savetxt(P.header + 'time_data.dat', time_output, header=time_header, delimiter=' '*3, fmt="%+.18e")

	##################################################
	# Frequency data save
	##################################################
	if P.split_current:
		freq_header = ("{:25s}" + " {:27s}"*24)\
			.format("f/f0",
					"Re[j_E_dir]", "Im[j_E_dir]", "Re[j_ortho]", "Im[j_ortho]",
					"I_E_dir", "I_ortho",
					"Re[j_intra_E_dir]", "Im[j_intra_E_dir]", "Re[j_intra_ortho]", "Im[j_intra_ortho]",
					"I_intra_E_dir", "I_intra_ortho",
					"Re[dtP_E_dir]", "Im[dtP_E_dir]", "Re[dtP_ortho]", "Im[dtP_ortho]",
					"I_dtP_E_dir", "I_dtP_ortho",
					"Re[j_intra_plus_dtP_E_dir]", "Im[j_intra_plus_dtP_E_dir]", "Re[j_intra_plus_dtP_ortho]", "Im[j_intra_plus_dtP_ortho]",
					"I_intra_plus_dtP_E_dir", "I_intra_plus_dtP_ortho")

		# Current same order as in time output, always real and imaginary part
		# next column -> corresponding intensities
		freq_output = np.column_stack([(W.freq/P.f).real,
									   W.j_E_dir.real, W.j_E_dir.imag, W.j_ortho.real, W.j_ortho.imag,
									   W.I_E_dir.real, W.I_ortho.real,
									   W.j_intra_E_dir.real, W.j_intra_E_dir.imag, W.j_intra_ortho.real, W.j_intra_ortho.imag,
									   W.I_intra_E_dir.real, W.I_intra_ortho.real,
									   W.dtP_E_dir.real, W.dtP_E_dir.imag, W.dtP_ortho.real, W.dtP_ortho.imag,
									   W.I_dtP_E_dir.real, W.I_dtP_ortho.real,
									   W.j_intra_plus_dtP_E_dir.real, W.j_intra_plus_dtP_E_dir.imag, W.j_intra_plus_dtP_ortho.real, W.j_intra_plus_dtP_ortho.imag,
									   W.I_intra_plus_dtP_E_dir.real, W.I_intra_plus_dtP_ortho.real,
										 W.j_deph_E_dir.real, W.j_deph_E_dir.imag, W.j_deph_ortho.real, W.j_deph_ortho.imag,
										 W.I_deph_E_dir.real, W.I_deph_ortho.real])
		if P.save_anom:
			for i in range(P.n):
				freq_header += (" {:27s}"*3).format(f"Re[j_anom_ortho[{i}]]", f"Im[j_anom_ortho[{i}]", \
											f"I_anom_ortho[{i}]")
				freq_output = np.column_stack((freq_output, W.j_anom_ortho[:, i].real, W.j_anom_ortho[:, i].imag, W.I_anom_ortho[:, i].real) )
			freq_header += (" {:27s}"*6).format("Re[j_anom_ortho]", "Im[j_anom_ortho]", "I_anom_ortho", "Re[j_intra_plus_anom_ortho]", "Im[j_intra_plus_anom_ortho]", "I_intra_plus_anom_ortho")
			freq_output = np.column_stack((freq_output, W.j_anom_ortho_full.real, W.j_anom_ortho_full.imag, W.I_anom_ortho_full, W.j_intra_plus_anom_ortho.real, W.j_intra_plus_anom_ortho.imag, W.I_intra_plus_anom_ortho.real))

	elif P.sheet_current:

		freq_header =("{:25s}").format('f/f0')
		freq_output = (W.freq/P.f).real

		for i in range(P.n_sheets):
			for j in range(P.n_sheets):
				freq_header += (" {:27s}"*6).format(f"Re[j_E_dir[{i},{j}]]", f"Im[j_E_dir[{i},{j}]]", \
					f"Re[j_ortho[{i},{j}]]", f"Im[j_ortho[{i},{j}]]", f"I_E_dir[{i},{j}]", f"I_ortho[{i},{j}]")
				freq_output = np.column_stack((freq_output, W.j_E_dir[:, i, j].real, W.j_E_dir[:, i, j].imag \
					,W.j_ortho[:, i, j].real, W.j_ortho[:, i, j].imag, W.I_E_dir[:, i, j].real, W.I_ortho[:, i, j].real) )

		freq_header += (" {:27s}"*6).format("Re[j_E_dir]", "Im[j_E_dir]", \
				"Re[j_ortho]", "Im[j_ortho]", "I_E_dir", "I_ortho")
		freq_output = np.column_stack((freq_output, W.j_E_dir_full.real, W.j_E_dir_full.imag \
				,W.j_ortho_full.real, W.j_ortho_full.imag, W.I_E_dir_full.real, W.I_ortho_full.real) )

	else:
		freq_header = ("{:25s}" + " {:27s}"*6)\
			.format("f/f0",
					"Re[j_E_dir]", "Im[j_E_dir]", "Re[j_ortho]", "Im[j_ortho]",
					"I_E_dir", "I_ortho")
		freq_output = np.column_stack([(W.freq/P.f).real,
									   W.j_E_dir.real, W.j_E_dir.imag, W.j_ortho.real, W.j_ortho.imag,
									   W.I_E_dir.real, W.I_ortho.real])


	# Make the maximum exponent double digits
	freq_output[np.abs(freq_output) <= 10e-100] = 0
	freq_output[np.abs(freq_output) >= 1e+100] = np.inf

	np.savetxt(P.header + 'frequency_data.dat', freq_output, header=freq_header, delimiter=' '*3, fmt="%+.18e")

	if P.gabor_transformation:
		for i in range(np.size(P.gabor_gaussian_center)):
			for j in range(np.size(P.gabor_window_width)):
				freq_header = ("{:25s}" + " {:27s}"*6)\
					.format("f/f0",
							"Re[j_E_dir]", "Im[j_E_dir]", "Re[j_ortho]", "Im[j_ortho]",
							"I_E_dir", "I_ortho")
				freq_output = np.column_stack([(W.freq/P.f).real,
											W.j_E_dir_GT[i][j].real, W.j_E_dir_GT[i][j].imag, W.j_ortho_GT[i][j].real, W.j_ortho_GT[i][j].imag,
											W.I_E_dir_GT[i][j].real, W.I_ortho_GT[i][j].real])
				# Make the maximum exponent double digits
				freq_output[np.abs(freq_output) <= 10e-100] = 0
				freq_output[np.abs(freq_output) >= 1e+100] = np.inf

				np.savetxt(f"gabor_trafo_center={(P.gabor_gaussian_center[i]*CoFa.au_to_fs):.4f}fs_width={(P.gabor_window_width[j]*CoFa.au_to_fs):.4f}fs_"+P.header\
					+ 'frequency_data.dat', freq_output, header=freq_header, delimiter=' '*3, fmt="%+.18e")

	if P.save_latex_pdf:
		write_and_compile_latex_PDF(T, W, P, sys, Mpi)


def write_efield_afield(T, P, W):

	time_header = ("{:25s}" + " {:27s}"*2).format("t", "E_field", "A_field")
	time_output = np.column_stack([T.t.real, T.E_field, T.A_field])

	# Make the maximum exponent double digits
	time_output[np.abs(time_output) <= 10e-100] = 0
	time_output[np.abs(time_output) >= 1e+100] = np.inf

	np.savetxt(P.header + 'fields_time_data.dat', time_output, header=time_header, delimiter=' '*3, fmt="%+.18e")

	freq_header = ("{:25s}" + " {:27s}"*4).format("f/f0", "Re[E_field]", "Im[E_field]", "Re[A_field]", "Im[A_field]")
	dt = T.t[1] - T.t[0]
	E_fourier = fourier(dt, T.E_field)
	A_fourier = fourier(dt, T.A_field)
	freq_output = np.column_stack([(W.freq/P.f).real, E_fourier.real, E_fourier.imag, A_fourier.real, A_fourier.imag])

	# Make the maximum exponent double digits
	freq_output[np.abs(freq_output) <= 10e-100] = 0
	freq_output[np.abs(freq_output) >= 1e+100] = np.inf

	np.savetxt(P.header + 'fields_frequency_data.dat', freq_output, header=freq_header, delimiter=' '*3, fmt="%+.18e")


def fourier_current_intensity(jt, window_function, dt_out, prefac_emission, freq, P):

	ndt_fft = freq.size
	ndt = np.size(jt, axis=0)

	jt_for_fft = np.zeros(ndt_fft, dtype=P.type_real_np)
	if np.ndim(jt) == 1:
		jt_for_fft[(ndt_fft - ndt)//2:(ndt_fft + ndt)//2] = jt[:]*window_function[:]
		jw = fourier(dt_out, jt_for_fft)
		Iw = prefac_emission*(freq**2)*np.abs(jw)**2

	elif np.ndim(jt) == 2:
		n       = np.size(jt, axis=1)
		jw      = np.empty([ndt_fft, n], dtype=P.type_complex_np)
		Iw      = np.empty([ndt_fft, n], dtype=P.type_real_np)

		for i in range(P.n):
			jt_for_fft[(ndt_fft - ndt)//2:(ndt_fft + ndt)//2] = jt[:, i]*window_function[:]
			jw[:, i] = fourier(dt_out, jt_for_fft)
			Iw[:, i] = prefac_emission*(freq**2)*np.abs(jw[:, i])**2

	elif np.ndim(jt) == 3:
		n       = np.size(jt, axis=1)
		jw      = np.empty([ndt_fft, n, n], dtype=P.type_complex_np)
		Iw      = np.empty([ndt_fft, n, n], dtype=P.type_real_np)

		for i in range(n):
			for j in range(n):
				jt_for_fft[(ndt_fft - ndt)//2:(ndt_fft + ndt)//2] = jt[:, i, j]*window_function[:]
				jw[:, i, j] = fourier(dt_out, jt_for_fft)
				Iw[:, i, j] = prefac_emission*(freq**2)*np.abs(jw[:, i, j])**2

	return Iw, jw


def print_user_info(P, B0=None, mu=None, incident_angle=None):
	"""
	Function that prints the input parameters if usr_info = True
	"""

	print("Input parameters:")
	print("Brillouin zone                  = " + P.BZ_type)
	print("Number of k-points              = " + str(P.Nk))
	print("Densiy matrix calculation method= " + str(P.dm_dynamics_method))
	if P.dm_dynamics_method in ('sbe', 'semiclassics'):
		print("ODE solver method               = " + str(P.solver_method))
		print("Order of k-derivative           = " + str(P.dk_order))
		print("Right hand side of ODE          = " + str(P.solver))
	print("Precision (default = double)    = " + str(P.precision))
	if P.BZ_type == 'hexagon':
		print("Driving field alignment         = " + P.align)
	elif P.BZ_type == 'rectangle':
		print("Driving field direction         = " + str(P.angle_inc_E_field))
	print("Pulse Frequency (THz)[a.u.]     = " + "("
		  + '{:.6f}'.format(P.f_THz) + ")"
		  + "[" + '{:.6f}'.format(P.f) + "]")
	if P.user_defined_field:
		print("User defined field parameters..")
	else:
		print("Driving amplitude (MV/cm)[a.u.] = " + "("
			+ '{:.6f}'.format(P.E0_MVpcm) + ")"
			+ "[" + '{:.6f}'.format(P.E0) + "]")
		print("Pulse Width (fs)[a.u.]          = " + "("
			+ '{:.6f}'.format(P.sigma_fs) + ")"
			+ "[" + '{:.6f}'.format(P.sigma) + "]")
		print("Chirp rate (THz)[a.u.]          = " + "("
			+ '{:.6f}'.format(P.chirp_THz) + ")"
			+ "[" + '{:.6f}'.format(P.chirp) + "]")
		print("Phase (1)[pi]                   = " + "("
			+ '{:.6f}'.format(P.phase) + ")"
			+ "[" + '{:.6f}'.format(P.phase/np.pi) + "]")
	print("Damping time (fs)[a.u.]         = " + "("
		  + '{:.6f}'.format(P.T2_fs) + ")"
		  + "[" + '{:.6f}'.format(P.T2) + "]")
	print("Total time (fs)[a.u.]           = " + "("
		  + '{:.6f}'.format(P.tf_fs - P.t0_fs) + ")"
		  + "[" + '{:.5f}'.format(P.tf - P.t0) + "]")
	print("Time step (fs)[a.u.]            = " + "("
		  + '{:.6f}'.format(P.dt_fs) + ")"
		  + "[" + '{:.6f}'.format(P.dt) + "]")
