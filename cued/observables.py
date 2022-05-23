import numpy as np
import numpy.linalg as lin
from cued.utility import ConversionFactors as co
from cued.utility import conditional_njit, evaluate_njit_matrix


##########################################################################################
### Observables for the 2-band code
##########################################################################################


##########################################################################################
## Observables working with density matrices that contain NO time data; only path
##########################################################################################
def make_polarization_path(path, P, sys):
	"""
	Function that calculates the polarization for the current path

	Parameters:
	-----------
	dipole : Symbolic Dipole
	pathlen : int
	    Length of one path
	n_time_steps : int
	    Number of time steps
	E_dir : np.ndarray [type_real_np]
	    Direction of the electric field
	A_field : np.ndarray [type_real_np]
	    Vector potential integrated from electric field
	gauge : string
	    System gauge 'length' or 'velocity'

	Returns:
	--------
	P_E_dir : np.ndarray [type_real_np]
	    Polarization in E-field direction
	P_ortho : np.ndarray [type_real_np]
	    Polarization orthogonal to E-field direction
	"""
	di_01xf = sys.Axfjit[0][1]
	di_01yf = sys.Ayfjit[0][1]

	E_dir = P.E_dir
	E_ort = P.E_ort

	kx_in_path_before_shift = path[:, 0]
	ky_in_path_before_shift = path[:, 1]
	pathlen = kx_in_path_before_shift.size

	type_complex_np = P.type_complex_np
	gauge = P.gauge
	@conditional_njit(type_complex_np)
	def polarization_path(solution, E_field, A_field):
		##################################################
		# Dipole container
		##################################################
		rho_cv = solution[:, 1, 0]

		d_01x = np.empty(pathlen, dtype=type_complex_np)
		d_01y = np.empty(pathlen, dtype=type_complex_np)

		d_E_dir = np.empty(pathlen, dtype=type_complex_np)
		d_ortho = np.empty(pathlen, dtype=type_complex_np)

		if gauge == 'length':
			kx_shift = 0
			ky_shift = 0
		if gauge == 'velocity':
			kx_shift = A_field*E_dir[0]
			ky_shift = A_field*E_dir[1]

		kx_in_path = kx_in_path_before_shift + kx_shift
		ky_in_path = ky_in_path_before_shift + ky_shift

		d_01x[:] = di_01xf(kx=kx_in_path, ky=ky_in_path)
		d_01y[:] = di_01yf(kx=kx_in_path, ky=ky_in_path)

		d_E_dir[:] = d_01x * E_dir[0] + d_01y * E_dir[1]
		d_ortho[:] = d_01x * E_ort[0] + d_01y * E_ort[1]

		P_E_dir = 2*np.real(np.sum(d_E_dir * rho_cv))
		P_ortho = 2*np.real(np.sum(d_ortho * rho_cv))

		return P_E_dir, P_ortho

	return polarization_path


def make_current_path(path, P, sys):
	'''
	Calculates the intraband current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
	where j_n(k) != (d/dk) E_n(k)

	Parameters
	----------
	sys : TwoBandSystem
	    Hamiltonian and related functions
	pathlen : int
	    Length of one path
	n_time_steps : int
	    Number of time steps
	E_dir : np.ndarray [type_real_np]
	    Direction of the electric field
	A_field : np.ndarray [type_real_np]
	    Vector potential integrated from electric field
	gauge : string
	    System gauge 'length' or 'velocity'

	Returns
	-------
	J_E_dir : np.ndarray [type_real_np]
	    intraband current j_intra in E-field direction
	J_ortho : np.ndarray [type_real_np]
	    intraband current j_intra orthogonal to E-field direction
	'''


	edxjit_v = sys.ederivfjit[0]
	edyjit_v = sys.ederivfjit[1]
	edxjit_c = sys.ederivfjit[2]
	edyjit_c = sys.ederivfjit[3]

	if P.save_anom:
		Bcurv_00 = sys.Bfjit[0][0]
		Bcurv_11 = sys.Bfjit[1][1]

	E_dir = P.E_dir
	E_ort = P.E_ort
	kx_in_path_before_shift = path[:, 0]
	ky_in_path_before_shift = path[:, 1]
	pathlen = kx_in_path_before_shift.size

	type_real_np = P.type_real_np
	type_complex_np = P.type_complex_np
	gauge = P.gauge
	save_anom = P.save_anom
	@conditional_njit(type_complex_np)
	def current_path(solution, E_field, A_field):

		rho_vv = solution[:, 0, 0]
		rho_cc = solution[:, 1, 1]
		##################################################
		# E derivative container
		##################################################
		edx_v = np.empty(pathlen, dtype=type_real_np)
		edy_v = np.empty(pathlen, dtype=type_real_np)
		edx_c = np.empty(pathlen, dtype=type_real_np)
		edy_c = np.empty(pathlen, dtype=type_real_np)

		e_deriv_E_dir_v = np.empty(pathlen, dtype=type_real_np)
		e_deriv_ortho_v = np.empty(pathlen, dtype=type_real_np)
		e_deriv_E_dir_c = np.empty(pathlen, dtype=type_real_np)
		e_deriv_ortho_c = np.empty(pathlen, dtype=type_real_np)

		if gauge == 'length':
			kx_shift = 0
			ky_shift = 0
			rho_vv_subs = 0
		if gauge == 'velocity':
			kx_shift = A_field*E_dir[0]
			ky_shift = A_field*E_dir[1]
			rho_vv_subs = 1

		kx_in_path = kx_in_path_before_shift + kx_shift
		ky_in_path = ky_in_path_before_shift + ky_shift

		edx_v[:] = edxjit_v(kx=kx_in_path, ky=ky_in_path)
		edy_v[:] = edyjit_v(kx=kx_in_path, ky=ky_in_path)
		edx_c[:] = edxjit_c(kx=kx_in_path, ky=ky_in_path)
		edy_c[:] = edyjit_c(kx=kx_in_path, ky=ky_in_path)


		e_deriv_E_dir_v[:] = edx_v * E_dir[0] + edy_v * E_dir[1]
		e_deriv_ortho_v[:] = edx_v * E_ort[0] + edy_v * E_ort[1]
		e_deriv_E_dir_c[:] = edx_c * E_dir[0] + edy_c * E_dir[1]
		e_deriv_ortho_c[:] = edx_c * E_ort[0] + edy_c * E_ort[1]

		J_E_dir = - np.sum(e_deriv_E_dir_v * (rho_vv.real - rho_vv_subs)) \
		          - np.sum(e_deriv_E_dir_c * rho_cc.real)
		J_ortho = - np.sum(e_deriv_ortho_v * (rho_vv.real - rho_vv_subs)) \
		          - np.sum(e_deriv_ortho_c * rho_cc.real)

		if save_anom:

			J_anom_ortho = np.zeros(2, dtype=type_real_np)

			Bcurv_v = np.empty(pathlen, dtype=type_complex_np)
			Bcurv_c = np.empty(pathlen, dtype=type_complex_np)

			Bcurv_v = Bcurv_00(kx=kx_in_path, ky=ky_in_path)
			Bcurv_c = Bcurv_11(kx=kx_in_path, ky=ky_in_path)

			J_anom_ortho[0] = -E_field * np.sum(Bcurv_v.real * rho_vv.real)
			J_anom_ortho[1] += -E_field * np.sum(Bcurv_c.real * rho_cc.real)
		else:
			J_anom_ortho = 0

		return J_E_dir, J_ortho, J_anom_ortho

	return current_path


def make_emission_exact_path_velocity(path, P, sys):
	"""
	Construct a function that calculates the emission for the system solution per path
	Works for velocity gauge.

	Parameters
	----------
	sys : TwoBandSystem
	    Hamiltonian and related functions
	path : np.ndarray [type_real_np]
	    kx and ky components of path
	E_dir : np.ndarray [type_real_np]
	    Direction of the electric field
	do_semicl : bool
	    if semiclassical calculation should be done
	curvature : SymbolicCurvature
	    Curvature is only needed for semiclassical calculation

	Returns
	-------
	emision_kernel : function
	    Calculates per timestep current of a path
	"""
	E_dir = P.E_dir

	hderivx = sys.hderivfjit[0]
	hdx_00 = hderivx[0][0]
	hdx_01 = hderivx[0][1]
	hdx_10 = hderivx[1][0]
	hdx_11 = hderivx[1][1]

	hderivy = sys.hderivfjit[1]
	hdy_00 = hderivy[0][0]
	hdy_01 = hderivy[0][1]
	hdy_10 = hderivy[1][0]
	hdy_11 = hderivy[1][1]

	Ujit = sys.Ujit
	U_00 = Ujit[0][0]
	U_01 = Ujit[0][1]
	U_10 = Ujit[1][0]
	U_11 = Ujit[1][1]

	Ujit_h = sys.Ujit_h
	U_h_00 = Ujit_h[0][0]
	U_h_01 = Ujit_h[0][1]
	U_h_10 = Ujit_h[1][0]
	U_h_11 = Ujit_h[1][1]

	if P.dm_dynamics_method == 'semiclassics':
		Bcurv_00 = sys.Bfjit[0][0]
		Bcurv_11 = sys.Bfjit[1][1]

	E_ort = np.array([E_dir[1], -E_dir[0]])

	kx_in_path_before_shift = path[:, 0]
	ky_in_path_before_shift = path[:, 1]
	pathlen = kx_in_path_before_shift.size

	type_complex_np = P.type_complex_np
	symmetric_insulator = P.symmetric_insulator
	dm_dynamics_method = P.dm_dynamics_method
	@conditional_njit(type_complex_np)
	def emission_exact_path_velocity(solution, E_field, A_field):
		'''
		Calculates current from the system density matrix

		Parameters:
		-----------
		solution : np.ndarray [type_complex_np]
		    Per timestep solution, idx 0 is k; idx 1 is fv, pvc, pcv, fc
		E_field : type_real_np
		    Per timestep E_field
		A_field : type_real_np
		    In the velocity gauge this determines the k-shift

		Returns:
		--------
		I_E_dir : type_real_np
		    Parallel to electric field component of current
		I_ortho : type_real_np
		    Orthogonal to electric field component of current
		'''
		solution = solution.reshape(pathlen, 4)
		##########################################################
		# H derivative container
		##########################################################
		h_deriv_x = np.empty((pathlen, 2, 2), dtype=type_complex_np)
		h_deriv_y = np.empty((pathlen, 2, 2), dtype=type_complex_np)
		h_deriv_E_dir = np.empty((pathlen, 2, 2), dtype=type_complex_np)
		h_deriv_ortho = np.empty((pathlen, 2, 2), dtype=type_complex_np)

		##########################################################
		# Wave function container
		##########################################################
		U = np.empty((pathlen, 2, 2), dtype=type_complex_np)
		U_h = np.empty((pathlen, 2, 2), dtype=type_complex_np)

		##########################################################
		# Berry curvature container
		##########################################################
		if dm_dynamics_method == 'semiclassics':
			Bcurv = np.empty((pathlen, 2), dtype=type_complex_np)

		kx_in_path = kx_in_path_before_shift + A_field*E_dir[0]
		ky_in_path = ky_in_path_before_shift + A_field*E_dir[1]

		h_deriv_x[:, 0, 0] = hdx_00(kx=kx_in_path, ky=ky_in_path)
		h_deriv_x[:, 0, 1] = hdx_01(kx=kx_in_path, ky=ky_in_path)
		h_deriv_x[:, 1, 0] = hdx_10(kx=kx_in_path, ky=ky_in_path)
		h_deriv_x[:, 1, 1] = hdx_11(kx=kx_in_path, ky=ky_in_path)

		h_deriv_y[:, 0, 0] = hdy_00(kx=kx_in_path, ky=ky_in_path)
		h_deriv_y[:, 0, 1] = hdy_01(kx=kx_in_path, ky=ky_in_path)
		h_deriv_y[:, 1, 0] = hdy_10(kx=kx_in_path, ky=ky_in_path)
		h_deriv_y[:, 1, 1] = hdy_11(kx=kx_in_path, ky=ky_in_path)

		h_deriv_E_dir[:, :, :] = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
		h_deriv_ortho[:, :, :] = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

		U[:, 0, 0] = U_00(kx=kx_in_path, ky=ky_in_path)
		U[:, 0, 1] = U_01(kx=kx_in_path, ky=ky_in_path)
		U[:, 1, 0] = U_10(kx=kx_in_path, ky=ky_in_path)
		U[:, 1, 1] = U_11(kx=kx_in_path, ky=ky_in_path)

		U_h[:, 0, 0] = U_h_00(kx=kx_in_path, ky=ky_in_path)
		U_h[:, 0, 1] = U_h_01(kx=kx_in_path, ky=ky_in_path)
		U_h[:, 1, 0] = U_h_10(kx=kx_in_path, ky=ky_in_path)
		U_h[:, 1, 1] = U_h_11(kx=kx_in_path, ky=ky_in_path)

		if dm_dynamics_method == 'semiclassics':
			Bcurv[:, 0] = Bcurv_00(kx=kx_in_path, ky=ky_in_path)
			Bcurv[:, 1] = Bcurv_11(kx=kx_in_path, ky=ky_in_path)

		I_E_dir = 0
		I_ortho = 0

		rho_vv = solution[:, 0]
		# rho_vc = solution[:, 1]
		rho_cv = solution[:, 2]
		rho_cc = solution[:, 3]

		if symmetric_insulator:
			rho_vv = -rho_cc + 1

		for i_k in range(pathlen):

			dH_U_E_dir = h_deriv_E_dir[i_k] @ U[i_k]
			U_h_H_U_E_dir = U_h[i_k] @ dH_U_E_dir

			dH_U_ortho = h_deriv_ortho[i_k] @ U[i_k]
			U_h_H_U_ortho = U_h[i_k] @ dH_U_ortho

			I_E_dir += - U_h_H_U_E_dir[0, 0].real * (rho_vv[i_k].real - 1)
			I_E_dir += - U_h_H_U_E_dir[1, 1].real * rho_cc[i_k].real
			I_E_dir += - 2*np.real(U_h_H_U_E_dir[0, 1] * rho_cv[i_k])

			I_ortho += - U_h_H_U_ortho[0, 0].real * (rho_vv[i_k].real - 1)
			I_ortho += - U_h_H_U_ortho[1, 1].real * rho_cc[i_k].real
			I_ortho += - 2*np.real(U_h_H_U_ortho[0, 1] * rho_cv[i_k])

			if dm_dynamics_method == 'semiclassics':
				# '-' because there is q^2 compared to q only at the SBE current
				I_ortho += -E_field * Bcurv[i_k, 0].real * rho_vv[i_k].real
				I_ortho += -E_field * Bcurv[i_k, 1].real * rho_cc[i_k].real

		return I_E_dir, I_ortho

	return emission_exact_path_velocity


def make_emission_exact_path_length(path, P, sys):
	"""
	Construct a function that calculates the emission for the system solution per path.
	Works for length gauge.

	Parameters
	----------
	sys : TwoBandSystem
	    Hamiltonian and related functions
	path : np.ndarray [type_real_np]
	    kx and ky components of path
	E_dir : np.ndarray [type_real_np]
	    Direction of the electric field
	do_semicl : bool
	    if semiclassical calculation should be done
	curvature : SymbolicCurvature
	    Curvature is only needed for semiclassical calculation

	Returns:
	--------
	emission_kernel : function
	    Calculates per timestep current of a path
	"""
	E_dir = P.E_dir
	E_ort = P.E_ort

	kx_in_path = path[:, 0]
	ky_in_path = path[:, 1]
	pathlen = kx_in_path.size

	##########################################################
	# Berry curvature container
	##########################################################
	if P.dm_dynamics_method == 'semiclassics':
		Bcurv = np.empty((pathlen, 2), dtype=P.type_complex_np)

	h_deriv_x = evaluate_njit_matrix(sys.hderivfjit[0], kx=kx_in_path, ky=ky_in_path,
	                                 dtype=P.type_complex_np)
	h_deriv_y = evaluate_njit_matrix(sys.hderivfjit[1], kx=kx_in_path, ky=ky_in_path,
	                                 dtype=P.type_complex_np)

	h_deriv_E_dir= h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
	h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

	U = evaluate_njit_matrix(sys.Ujit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
	U_h = evaluate_njit_matrix(sys.Ujit_h, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

	if P.dm_dynamics_method == 'semiclassics':
		Bcurv[:, 0] = sys.Bfjit[0][0](kx=kx_in_path, ky=ky_in_path)
		Bcurv[:, 1] = sys.Bfjit[1][1](kx=kx_in_path, ky=ky_in_path)

	symmetric_insulator = P.symmetric_insulator
	dm_dynamics_method = P.dm_dynamics_method
	@conditional_njit(P.type_complex_np)
	def emission_exact_path_length(solution, E_field, A_field):
		'''
		Parameters:
		-----------
		solution : np.ndarray [type_complex_np]
		    Per timestep solution, idx 0 is k; idx 1 is fv, pvc, pcv, fc
		E_field : type_real_np
		    Per timestep E_field
		_A_field : dummy
		    In the length gauge this is just a dummy variable

		Returns:
		--------
		I_E_dir : type_real_np
		    Parallel to electric field component of current
		I_ortho : type_real_np
		    Orthogonal to electric field component of current
		'''
		solution = solution.reshape(pathlen, 4)

		I_E_dir = 0
		I_ortho = 0

		rho_vv = solution[:, 0]
		rho_vc = solution[:, 1]
		rho_cv = solution[:, 2]
		rho_cc = solution[:, 3]

		if symmetric_insulator:
			rho_vv = -rho_cc

		for i_k in range(pathlen):

			dH_U_E_dir = h_deriv_E_dir[i_k] @ U[i_k]
			U_h_H_U_E_dir = U_h[i_k] @ dH_U_E_dir

			dH_U_ortho = h_deriv_ortho[i_k] @ U[i_k]
			U_h_H_U_ortho = U_h[i_k] @ dH_U_ortho

			I_E_dir += - U_h_H_U_E_dir[0, 0].real * rho_vv[i_k].real
			I_E_dir += - U_h_H_U_E_dir[1, 1].real * rho_cc[i_k].real
			I_E_dir += - 2*np.real(U_h_H_U_E_dir[0, 1] * rho_cv[i_k])

			I_ortho += - U_h_H_U_ortho[0, 0].real * rho_vv[i_k].real
			I_ortho += - U_h_H_U_ortho[1, 1].real * rho_cc[i_k].real
			I_ortho += - 2*np.real(U_h_H_U_ortho[0, 1] * rho_cv[i_k])

			if dm_dynamics_method == 'semiclassics':
				# '-' because there is q^2 compared to q only at the SBE current
				I_ortho += -E_field * Bcurv[i_k, 0].real * rho_vv[i_k].real
				I_ortho += -E_field * Bcurv[i_k, 1].real * rho_vc[i_k].real

		return I_E_dir, I_ortho

	return emission_exact_path_length

##########################################################################################
### Observables from given bandstructures
##########################################################################################
def make_current_exact_bandstructure(path, P, sys):

	Nk1 = P.Nk1
	n = P.n

	kx_in_path = path[:, 0]
	ky_in_path = path[:, 1]

	mel_x = evaluate_njit_matrix(sys.melxjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
	mel_y = evaluate_njit_matrix(sys.melyjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

	mel_in_path = P.E_dir[0]*mel_x + P.E_dir[1]*mel_y
	mel_ortho = P.E_ort[0]*mel_x + P.E_ort[1]*mel_y

	@conditional_njit(P.type_complex_np)
	def current_exact_path(solution, _E_field=0, _A_field=0):

		J_exact_E_dir = 0
		J_exact_ortho = 0

		for i_k in range(Nk1):
			for i in range(n):
				J_exact_E_dir -=  mel_in_path[i_k, i, i].real * solution[i_k, i, i].real
				J_exact_ortho -=  mel_ortho[i_k, i, i].real * solution[i_k, i, i].real
				for j in range(n):
					if i != j:
						J_exact_E_dir -= np.real(mel_in_path[i_k, i, j] * solution[i_k, j, i])
						J_exact_ortho -= np.real(mel_ortho[i_k, i, j] * solution[i_k, j, i])

		return J_exact_E_dir, J_exact_ortho
	return current_exact_path


def make_intraband_current_bandstructure(path, P, sys):
	"""
	    Function that calculates the intraband current from eq. (76 and 77) with or without the
	    	anomalous contribution via the Berry curvature
	"""

	Nk1 = P.Nk1
	n = P.n
	save_anom = P.save_anom

	kx_in_path = path[:, 0]
	ky_in_path = path[:, 1]

	ederivx = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)
	ederivy = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)

	for i in range(P.n):
		ederivx[:, i] = sys.dkxejit[i](kx=kx_in_path, ky=ky_in_path)
		ederivy[:, i] = sys.dkyejit[i](kx=kx_in_path, ky=ky_in_path)

	ederiv_in_path = P.E_dir[0]*ederivx + P.E_dir[1]*ederivy
	ederiv_ortho = P.E_ort[0]*ederivx + P.E_ort[1]*ederivy

	@conditional_njit(P.type_complex_np)
	def current_intra_path(solution, E_field, A_field):

		J_intra_E_dir = 0
		J_intra_ortho = 0
		J_anom_ortho = 0

		for k in range(Nk1):
			for i in range(n):
				J_intra_E_dir -= ederiv_in_path[k, i] * solution[k, i, i].real
				J_intra_ortho -= ederiv_ortho[k, i] * solution[k, i, i].real

				if save_anom:
					J_anom_ortho += 0
					print('J_anom not implemented')

		return J_intra_E_dir, J_intra_ortho, J_anom_ortho
	return current_intra_path


def make_polarization_inter_bandstructure(P, sys):
	"""
	    Function that calculates the interband polarization from eq. (74)
	"""
	dipole_in_path = sys.dipole_in_path
	dipole_ortho = sys.dipole_ortho
	n = P.n
	Nk1 = P.Nk1
	type_complex_np = P.type_complex_np

	@conditional_njit(type_complex_np)
	def polarization_inter_path(solution, E_field, A_field):

		P_inter_E_dir = 0
		P_inter_ortho = 0

		for k in range(Nk1):
			for i in range(n):
				for j in range(n):
					if i > j:
						P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])
						P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])
		return P_inter_E_dir, P_inter_ortho
	return polarization_inter_path

##########################################################################################
### Observables for the n-band code
##########################################################################################


def make_current_exact_path_hderiv_length(path, P, sys):

	"""
	    	Function that calculates the exact current via eq. (79)
	"""

	E_dir = P.E_dir
	E_ort = P.E_ort

	Nk1 = P.Nk1
	n = P.n
	n_sheets = P.n_sheets
	type_complex_np = P.type_complex_np
	type_real_np = P.type_real_np
	sheet_current= P.sheet_current
	dm_dynamics_method = P.dm_dynamics_method
	wf_in_path = sys.wf_in_path
	Bcurv_path = sys.Bcurv_path

	kx = path[:, 0]
	ky = path[:, 1]

	dhdkx = evaluate_njit_matrix(sys.hderivfjit[0], kx=kx, ky=ky, dtype=type_complex_np)
	dhdky = evaluate_njit_matrix(sys.hderivfjit[1], kx=kx, ky=ky, dtype=type_complex_np)

	matrix_element_x = np.zeros([Nk1, n, n], dtype=type_complex_np)
	matrix_element_y = np.zeros([Nk1, n, n], dtype=type_complex_np)

	if P.sheet_current:
		matrix_element_x = dhdkx
		matrix_element_y = dhdky

	elif P.dm_dynamics_method == 'semiclassics':
		for i in range(n):
			matrix_element_x[:, i, i] = sys.ederivx_path[:, i]
			matrix_element_y[:, i, i] = sys.ederivy_path[:, i]

	else:
		for i_k in range(Nk1):
			buff = dhdkx[i_k, :, :] @ wf_in_path[i_k, :, :]
			matrix_element_x[i_k, :, :] = np.conjugate(wf_in_path[i_k, :, :].T) @ buff

			buff = dhdky[i_k, :, :] @ wf_in_path[i_k, :, :]
			matrix_element_y[i_k, :, :] = np.conjugate(wf_in_path[i_k, :, :].T) @ buff


	mel_in_path = matrix_element_x * E_dir[0] + matrix_element_y * E_dir[1]
	mel_ortho = matrix_element_x * E_ort[0] + matrix_element_y * E_ort[1]


	@conditional_njit(type_complex_np)
	def current_exact_path_hderiv_length(solution, E_field, _A_field):

		if sheet_current:

			J_exact_E_dir = np.zeros((n_sheets, n_sheets), dtype=type_real_np)
			J_exact_ortho = np.zeros((n_sheets, n_sheets), dtype=type_real_np)

			n_s = int(n/n_sheets)
			for i_k in range(Nk1):
				sol = np.zeros((n,n), dtype=type_complex_np)
				for z in range(n):
					for zprime in range(n):
						for n_i in range(n):
							for n_j in range(n):
								sol[z, zprime] += np.conjugate(wf_in_path[i_k, z, n_i])*wf_in_path[i_k, zprime, n_j]*solution[i_k, n_j, n_i]
				for s_i in range(n_sheets):
					for s_j in range(n_sheets):
						for i in range(n_s):
							for j in range(n_s):
									J_exact_E_dir[s_i, s_j] -= np.real(mel_in_path[i_k, n_s*s_i + i, n_s*s_j + j] * sol[n_s*s_i + i, n_s*s_j + j])
									J_exact_ortho[s_i, s_j] -= np.real(mel_ortho[i_k, n_s*s_i + i, n_s*s_j + j] * sol[n_s*s_i + i, n_s*s_j + j])

		else:
			J_exact_E_dir = 0
			J_exact_ortho = 0

			for i_k in range(Nk1):
				for i in range(n):
					J_exact_E_dir -= mel_in_path[i_k, i, i].real * solution[i_k, i, i].real
					J_exact_ortho -= mel_ortho[i_k, i, i].real * solution[i_k, i, i].real
					for j in range(n):
						if i != j:
							J_exact_E_dir -= np.real(mel_in_path[i_k, i, j] * solution[i_k, j, i])
							J_exact_ortho -= np.real(mel_ortho[i_k, i, j] * solution[i_k, j, i])

					if dm_dynamics_method == 'semiclassics':
						J_exact_ortho -= E_field * Bcurv_path[i_k, i].real * solution[i_k, i, i].real

		return J_exact_E_dir, J_exact_ortho
	return current_exact_path_hderiv_length


def make_polarization_inter_path_length(P, sys):
	"""
	    Function that calculates the interband polarization from eq. (74)
	"""
	dipole_in_path = sys.dipole_in_path
	dipole_ortho = sys.dipole_ortho
	n = P.n
	Nk1 = P.Nk1
	type_complex_np = P.type_complex_np

	@conditional_njit(type_complex_np)
	def polarization_inter_path_length(solution, E_field, A_field):

		P_inter_E_dir = 0
		P_inter_ortho = 0

		for k in range(Nk1):
			for i in range(n):
				for j in range(n):
					if i > j:
						P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])
						P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])
		return P_inter_E_dir, P_inter_ortho
	return polarization_inter_path_length


def make_intraband_current_path_length(path, P, sys):
	"""
	    Function that calculates the intraband current from eq. (76 and 77) with or without the
	    anomalous contribution via the Berry curvature
	"""
	E_dir = P.E_dir
	E_ort = P.E_ort

	Nk1 = P.Nk1
	n = P.n
	epsilon = P.epsilon
	type_complex_np = P.type_complex_np
	type_real_np = P.type_real_np
	save_anom = P.save_anom
	Bcurv_path = sys.Bcurv_path

	# derivative of band structure

	pathplusx = np.copy(path)
	pathplusx[:, 0] += epsilon
	pathminusx = np.copy(path)
	pathminusx[:, 0] -= epsilon
	pathplusy = np.copy(path)
	pathplusy[:, 1] += epsilon
	pathminusy = np.copy(path)
	pathminusy[:, 1] -= epsilon

	pathplus2x = np.copy(path)
	pathplus2x[:, 0] += 2*epsilon
	pathminus2x = np.copy(path)
	pathminus2x[:, 0] -= 2*epsilon
	pathplus2y = np.copy(path)
	pathplus2y[:, 1] += 2*epsilon
	pathminus2y = np.copy(path)
	pathminus2y[:, 1] -= 2*epsilon

	eplusx, wfplusx = sys.diagonalize_path(pathplusx, P)
	eminusx, wfminusx = sys.diagonalize_path(pathminusx, P)
	eplusy, wfplusy = sys.diagonalize_path(pathplusy, P)
	eminusy, wfminusy = sys.diagonalize_path(pathminusy, P)

	eplus2x, wfplus2x = sys.diagonalize_path(pathplus2x, P)
	eminus2x, wfminus2x = sys.diagonalize_path(pathminus2x, P)
	eplus2y, wfplus2y = sys.diagonalize_path(pathplus2y, P)
	eminus2y, wfminus2y = sys.diagonalize_path(pathminus2y, P)

	ederivx = (-eplus2x + 8*eplusx - 8*eminusx + eminus2x)/(12*epsilon)
	ederivy = (-eplus2y + 8*eplusy - 8*eminusy + eminus2y)/(12*epsilon)

	# In E-field direction and orthogonal

	ederiv_in_path = E_dir[0] * ederivx + E_dir[1] * ederivy
	ederiv_ortho = E_ort[0] * ederivx + E_ort[1] * ederivy

	@conditional_njit(type_complex_np)
	def current_intra_path_length(solution, E_field=0, A_field=0):

		J_intra_E_dir = 0
		J_intra_ortho = 0
		J_anom_ortho = np.zeros(n, dtype=type_real_np)

		for k in range(Nk1):
			for i in range(n):
				J_intra_E_dir -= ederiv_in_path[k, i] * solution[k, i, i].real
				J_intra_ortho -= ederiv_ortho[k, i] * solution[k, i, i].real

				if save_anom:
					J_anom_ortho[i] -= E_field * Bcurv_path[k, i].real * solution[k, i, i].real

		return J_intra_E_dir, J_intra_ortho, J_anom_ortho
	return current_intra_path_length


######## VELOCITY GAUGE ###########

def make_current_exact_path_hderiv_velocity(path, P, sys):

	"""
	    Function that calculates the exact current via eq. (79)
	"""

	E_dir = P.E_dir
	E_ort = P.E_ort

	Nk1 = P.Nk1
	n = P.n
	n_sheets = P.n_sheets
	type_complex_np = P.type_complex_np
	type_real_np = P.type_real_np
	sheet_current= P.sheet_current
	dm_dynamics_method = P.dm_dynamics_method
	degenerate_eigenvalues = sys.degenerate_eigenvalues
	gidx = P.gidx
	sheet_current = P.sheet_current
	epsilon = P.epsilon

	kx_before_shift = path[:, 0]
	ky_before_shift = path[:, 1]
	pathlen = kx_before_shift.size


	def current_exact_path_hderiv_velocity(solution, E_field, A_field):

		kx_in_path = kx_before_shift + A_field*E_dir[0]
		ky_in_path = ky_before_shift + A_field*E_dir[1]

		dhdkx = evaluate_njit_matrix(sys.hderivfjit[0], kx=kx_in_path, ky=ky_in_path, dtype=type_complex_np)
		dhdky = evaluate_njit_matrix(sys.hderivfjit[1], kx=kx_in_path, ky=ky_in_path, dtype=type_complex_np)

		h_in_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		if dm_dynamics_method == 'semiclassics':
			hpex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp5ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm5ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp5ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm5ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hpexm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hmexm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp2expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp3expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp4expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm2expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm3expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm4expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp2exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp3exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hp4exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm2exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm3exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
			hm4exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)

		for i in range(n):
			for j in range(n):
				for k in range(pathlen):
					kx = kx_in_path[k]
					ky = ky_in_path[k]
					h_in_path[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky)


		for i in range(n):
			for j in range(n):
				for k in range(pathlen):
					kx = kx_in_path[k]
					ky = ky_in_path[k]
					h_in_path[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky)
					if dm_dynamics_method == 'semiclassics':
						hpex[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky)
						hmex[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky)
						hpey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+P.epsilon)
						hmey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-P.epsilon)
						hp2ex[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky)
						hm2ex[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky)
						hp2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+2*P.epsilon)
						hm2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-2*P.epsilon)
						hp3ex[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky)
						hm3ex[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky)
						hp3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+3*P.epsilon)
						hm3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-3*P.epsilon)
						hp4ex[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky)
						hm4ex[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky)
						hp4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+4*P.epsilon)
						hm4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-4*P.epsilon)
						hp5ex[k, i, j] = sys.hfjit[i][j](kx=kx+5*P.epsilon, ky=ky)
						hm5ex[k, i, j] = sys.hfjit[i][j](kx=kx-5*P.epsilon, ky=ky)
						hp5ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+5*P.epsilon)
						hm5ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-5*P.epsilon)
						hpexpey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+P.epsilon)
						hpexp2ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+2*P.epsilon)
						hpexp3ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+3*P.epsilon)
						hpexp4ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+4*P.epsilon)
						hpexmey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-P.epsilon)
						hpexm2ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-2*P.epsilon)
						hpexm3ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-3*P.epsilon)
						hpexm4ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-4*P.epsilon)
						hmexpey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+P.epsilon)
						hmexp2ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+2*P.epsilon)
						hmexp3ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+3*P.epsilon)
						hmexp4ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+4*P.epsilon)
						hmexmey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-P.epsilon)
						hmexm2ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-2*P.epsilon)
						hmexm3ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-3*P.epsilon)
						hmexm4ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-4*P.epsilon)
						hp2expey[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky+P.epsilon)
						hp3expey[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky+P.epsilon)
						hp4expey[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky+P.epsilon)
						hm2expey[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky+P.epsilon)
						hm3expey[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky+P.epsilon)
						hm4expey[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky+P.epsilon)
						hp2exmey[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky-P.epsilon)
						hp3exmey[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky-P.epsilon)
						hp4exmey[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky-P.epsilon)
						hm2exmey[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky-P.epsilon)
						hm3exmey[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky-P.epsilon)
						hm4exmey[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky-P.epsilon)

		buf, wf_in_path = diagonalize_h(h_in_path)

		if dm_dynamics_method == 'semiclassics':
			_buf, _buf, ederivx, ederivy = derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
																	hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

			Bcurv_path = __berry_curvature(h_in_path, hpex, hmex, hpey, hmey, hp2ex, hm2ex, hp2ey, hm2ey, hp3ex, hm3ex, hp3ey, hm3ey, hp4ex, hm4ex, \
					hp4ey, hm4ey, hp5ex, hm5ex, hp5ey, hm5ey, hpexpey, hpexp2ey, hpexp3ey, hpexp4ey, hpexmey, hpexm2ey, hpexm3ey, hpexm4ey, \
					hmexpey, hmexp2ey, hmexp3ey, hmexp4ey, hmexmey, hmexm2ey, hmexm3ey, hmexm4ey, hp2expey, hp3expey, hp4expey, hm2expey, \
					hm3expey, hm4expey, hp2exmey, hp3exmey, hp4exmey, hm2exmey, hm3exmey, hm4exmey)

			J_exact_E_dir, J_exact_ortho = calculate_current(dhdkx, dhdky, wf_in_path, solution, ederivx=ederivx, ederivy=ederivy, bcurv=Bcurv_path, E_field=E_field)            
		
		else:
			J_exact_E_dir, J_exact_ortho = calculate_current(dhdkx, dhdky, wf_in_path, solution)

		return J_exact_E_dir, J_exact_ortho

	@conditional_njit(type_complex_np)
	def derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = hpex[:, 0, 0].size

		xderivative = np.empty((pathlen, n, n), dtype=type_complex_np)
		yderivative = np.empty((pathlen, n, n), dtype=type_complex_np)

		eplusx, wfplusx = diagonalize_h(hpex)
		eminusx, wfminusx = diagonalize_h(hmex)
		eplusy, wfplusy = diagonalize_h(hpey)
		eminusy, wfminusy = diagonalize_h(hmey)

		eplus2x, wfplus2x = diagonalize_h(hp2ex)
		eminus2x, wfminus2x = diagonalize_h(hm2ex)
		eplus2y, wfplus2y = diagonalize_h(hp2ey)
		eminus2y, wfminus2y = diagonalize_h(hm2ey)

		eplus3x, wfplus3x = diagonalize_h(hp3ex)
		eminus3x, wfminus3x = diagonalize_h(hm3ex)
		eplus3y, wfplus3y = diagonalize_h(hp3ey)
		eminus3y, wfminus3y = diagonalize_h(hm3ey)

		eplus4x, wfplus4x = diagonalize_h(hp4ex)
		eminus4x, wfminus4x = diagonalize_h(hm4ex)
		eplus4y, wfplus4y = diagonalize_h(hp4ey)
		eminus4y, wfminus4y = diagonalize_h(hm4ey)

		xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
		yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
		ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
		ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

		return xderivative, yderivative, ederivx, ederivy

	@conditional_njit(type_complex_np)
	def __berry_curvature(h_in_path, hpex, hmex, hpey, hmey, hp2ex, hm2ex, hp2ey, hm2ey, hp3ex, hm3ex, hp3ey, hm3ey, hp4ex, hm4ex, \
					hp4ey, hm4ey, hp5ex, hm5ex, hp5ey, hm5ey, hpexpey, hpexp2ey, hpexp3ey, hpexp4ey, hpexmey, hpexm2ey, hpexm3ey, hpexm4ey, \
					hmexpey, hmexp2ey, hmexp3ey, hmexp4ey, hmexmey, hmexm2ey, hmexm3ey, hmexm4ey, hp2expey, hp3expey, hp4expey, hm2expey, \
					hm3expey, hm4expey, hp2exmey, hp3exmey, hp4exmey, hm2exmey, hm3exmey, hm4exmey):

		pathlen = path[:, 0].size

		dAydx = np.empty((pathlen, n, n), dtype=type_complex_np)
		dAxdy = np.empty((pathlen, n, n), dtype=type_complex_np)

		Ax_plusx, Ay_plusx = dipole_path(hpex, hp2ex, h_in_path, hp3ex, hmex, hp4ex, hm2ex, hp5ex, hm3ex, \
		                                 hpexpey, hpexmey, hpexp2ey, hpexm2ey, hpexp3ey, hpexm3ey, hpexp4ey, hpexm4ey)
		Ax_minusx, Ay_minusx = dipole_path(hmex, h_in_path, hm2ex, hpex, hm3ex, hp2ex, hm4ex, hp3ex, hm5ex, \
		                                   hmexpey, hmexmey, hmexp2ey, hmexm2ey, hmexp3ey, hmexm3ey, hmexp4ey, hmexm4ey)
		Ax_plusy, Ay_plusy = dipole_path(hpey, hpexpey, hmexpey, hp2expey, hm2expey, hp3expey, hm3expey, hp4expey, hm4expey, \
		                                 hp2ey, h_in_path, hp3ey, hmey, hp4ey, hm2ey, hp5ey, hm3ey)
		Ax_minusy, Ay_minusy = dipole_path(hmey, hpexmey, hmexmey, hp2exmey, hm2exmey, hp3exmey, hm3exmey, hp4exmey, hm4exmey, \
		                                   h_in_path, hm2ey, hpey, hm3ey, hp2ey, hm4ey, hp3ey, hm5ey)

		dAxdy = (Ax_plusy - Ax_minusy)/(2*epsilon)
		dAydx = (Ay_plusx - Ay_minusx)/(2*epsilon)

		Bcurv = np.zeros((pathlen, n), dtype=type_complex_np)

		for i in range(n):
			for i_k in range(pathlen):
				Bcurv[i_k, i] = dAxdy[i_k, i, i] - dAydx[i_k, i, i]

		return Bcurv

	@conditional_njit(type_complex_np)
	def dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = h_in_path[:, 0, 0].size

		dx_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		dy_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		

		_buf, wf_path = diagonalize_h(h_in_path)
		dwfkx_path, dwfky_path, _buf, _buf = derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
		                                                     hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

		for i in range(pathlen):
			dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
			dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

		return dx_path, dy_path


	@conditional_njit(type_complex_np)
	def diagonalize_h(h_in_path):

		e_path = np.empty((pathlen, n), dtype=type_real_np)
		wf_path = np.empty((pathlen, n, n), dtype=type_complex_np)

		for i in range(pathlen):
			e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
			if degenerate_eigenvalues:
				for j in range(int(n/2)):
					wf1 = np.copy(wf_buff[:, 2*j])
					wf2 = np.copy(wf_buff[:, 2*j+1])
					wf_buff[:, 2*j] *= wf2[n-2]
					wf_buff[:, 2*j] -= wf1[n-2]*wf2
					wf_buff[:, 2*j+1] *= wf1[n-1]
					wf_buff[:, 2*j+1] -= wf2[n-1]*wf1
			wf_gauged_entry = np.copy(wf_buff[gidx, :])
			wf_buff[gidx, :] = np.abs(wf_gauged_entry)
			wf_buff[~(np.arange(n) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
			wf_path[i] = wf_buff

		return e_path, wf_path


	@conditional_njit(type_complex_np)
	def calculate_current(dhdkx, dhdky, wf_in_path, solution, ederivx=0, ederivy=0, bcurv=0, E_field=0):

		matrix_element_x = np.zeros((Nk1, n, n), dtype=type_complex_np)
		matrix_element_y = np.zeros((Nk1, n, n), dtype=type_complex_np)

		if sheet_current:
			matrix_element_x = dhdkx
			matrix_element_y = dhdky

		elif dm_dynamics_method == 'semiclassics':
			for i in range(n):
				matrix_element_x[:, i, i] = ederivx[:, i]
				matrix_element_y[:, i, i] = ederivy[:, i]

		else:
			for i_k in range(Nk1):
				buff = dhdkx[i_k, :, :] @ wf_in_path[i_k, :, :]
				matrix_element_x[i_k, :, :] = (np.conjugate(wf_in_path[i_k, :, :].T) @ buff).T

				buff = dhdky[i_k, :, :] @ wf_in_path[i_k,:,:]
				matrix_element_y[i_k, :, :] = (np.conjugate(wf_in_path[i_k, :, :].T) @ buff).T


		mel_in_path = matrix_element_x * E_dir[0] + matrix_element_y * E_dir[1]
		mel_ortho = matrix_element_x * E_ort[0] + matrix_element_y * E_ort[1]

		if sheet_current: # CHECK THIS FOR VELOCITY GAUGE!!!

			J_exact_E_dir = np.zeros((n_sheets, n_sheets), dtype=type_real_np)
			J_exact_ortho = np.zeros((n_sheets, n_sheets), dtype=type_real_np)

			n_s = int(n/n_sheets)
			for i_k in range(Nk1):
				sol = np.zeros((n,n), dtype=type_complex_np)
				for z in range(n):
					for zprime in range(n):
						for n_i in range(n):
							for n_j in range(n):
								sol[z, zprime] += np.conjugate(wf_in_path[i_k, z, n_i])*wf_in_path[i_k, zprime, n_j]*solution[i_k, n_j, n_i]
				for s_i in range(n_sheets):
					for s_j in range(n_sheets):
						for i in range(n_s):
							for j in range(n_s):
									J_exact_E_dir[s_i, s_j] += - np.real( mel_in_path[i_k, n_s*s_i + i, n_s*s_j + j] * sol[n_s*s_i + i, n_s*s_j + j] )
									J_exact_ortho[s_i, s_j] += - np.real( mel_ortho[i_k, n_s*s_i + i, n_s*s_j + j] * sol[n_s*s_i + i, n_s*s_j + j] )

		else:
			J_exact_E_dir = 0
			J_exact_ortho = 0

			for i_k in range(Nk1):
				for i in range(n):
					if i == 0:
						J_exact_E_dir += - ( mel_in_path[i_k, i, i].real * ( solution[i_k, i, i].real - 1 ) )
						J_exact_ortho += - ( mel_ortho[i_k, i, i].real * ( solution[i_k, i, i].real - 1 ) )
					else:
						J_exact_E_dir += - ( mel_in_path[i_k, i, i].real * solution[i_k, i, i].real )
						J_exact_ortho += - ( mel_ortho[i_k, i, i].real * solution[i_k, i, i].real )
					for j in range(n):
						if i != j and j>i:
							J_exact_E_dir += - 2*np.real( mel_in_path[i_k, i, j] * solution[i_k, j, i] )
							J_exact_ortho += - 2*np.real( mel_ortho[i_k, i, j] * solution[i_k, j, i] )

					if dm_dynamics_method == 'semiclassics':
						J_exact_ortho += - E_field * bcurv[i_k, i].real * solution[i_k, i, i].real

		return J_exact_E_dir, J_exact_ortho

	return current_exact_path_hderiv_velocity


def make_polarization_inter_path_velocity(path, P, sys):
	"""
	    Function that calculates the interband polarization from eq. (74)
	"""
	dipole_in_path = sys.dipole_in_path
	dipole_ortho = sys.dipole_ortho
	n = P.n
	Nk1 = P.Nk1
	type_complex_np = P.type_complex_np
	type_real_np = P.type_real_np
	E_dir = P.E_dir
	E_ort = P.E_ort
	epsilon = P.epsilon
	gidx = P.gidx
	degenerate_eigenvalues = sys.degenerate_eigenvalues
	dm_dynamics_method = P.dm_dynamics_method

	kx_before_shift = path[:, 0]
	ky_before_shift = path[:, 1]
	pathlen = kx_before_shift.size

	def polarization_inter_path_velocity(solution, E_field, A_field):

		kx_in_path = kx_before_shift + A_field*E_dir[0]
		ky_in_path = ky_before_shift + A_field*E_dir[1]
		pathlen = kx_before_shift.size

		h_in_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)

		for i in range(n):
			for j in range(n):
				for k in range(pathlen):
					kx = kx_in_path[k]
					ky = ky_in_path[k]
					h_in_path[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky)
					hpex[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky)
					hmex[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky)
					hpey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+P.epsilon)
					hmey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-P.epsilon)
					hp2ex[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky)
					hm2ex[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky)
					hp2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+2*P.epsilon)
					hm2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-2*P.epsilon)
					hp3ex[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky)
					hm3ex[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky)
					hp3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+3*P.epsilon)
					hm3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-3*P.epsilon)
					hp4ex[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky)
					hm4ex[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky)
					hp4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+4*P.epsilon)
					hm4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-4*P.epsilon)

		# calculate dipole
		e_in_path, wf_in_path = diagonalize_path(h_in_path)
		dipole_path_x, dipole_path_y = dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
		                                           hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

		dipole_in_path = E_dir[0]*dipole_path_x + E_dir[1]*dipole_path_y
		dipole_ortho = E_ort[0]*dipole_path_x + E_ort[1]*dipole_path_y

		# calculate polarization
		P_inter_E_dir, P_inter_ortho = calculate_polarization(dipole_in_path, dipole_ortho, solution)

		return P_inter_E_dir, P_inter_ortho

	@conditional_njit(type_complex_np)
	def calculate_polarization(dipole_in_path, dipole_ortho, solution):

		P_inter_E_dir = 0
		P_inter_ortho = 0

		for k in range(Nk1):
			for i in range(n):
				for j in range(n):
					if i > j:
						P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])
						P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])

		return P_inter_E_dir, P_inter_ortho

	@conditional_njit(type_complex_np)
	def __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = hpex[:, 0, 0].size

		xderivative = np.empty((pathlen, n, n), dtype=type_complex_np)
		yderivative = np.empty((pathlen, n, n), dtype=type_complex_np)

		eplusx, wfplusx = diagonalize_path(hpex)
		eminusx, wfminusx = diagonalize_path(hmex)
		eplusy, wfplusy = diagonalize_path(hpey)
		eminusy, wfminusy = diagonalize_path(hmey)

		eplus2x, wfplus2x = diagonalize_path(hp2ex)
		eminus2x, wfminus2x = diagonalize_path(hm2ex)
		eplus2y, wfplus2y = diagonalize_path(hp2ey)
		eminus2y, wfminus2y = diagonalize_path(hm2ey)

		eplus3x, wfplus3x = diagonalize_path(hp3ex)
		eminus3x, wfminus3x = diagonalize_path(hm3ex)
		eplus3y, wfplus3y = diagonalize_path(hp3ey)
		eminus3y, wfminus3y = diagonalize_path(hm3ey)

		eplus4x, wfplus4x = diagonalize_path(hp4ex)
		eminus4x, wfminus4x = diagonalize_path(hm4ex)
		eplus4y, wfplus4y = diagonalize_path(hp4ey)
		eminus4y, wfminus4y = diagonalize_path(hm4ey)

		xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
		yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
		ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
		ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

		return xderivative, yderivative, ederivx, ederivy

	@conditional_njit(type_complex_np)
	def diagonalize_path(h_in_path):

		pathlen = h_in_path[:, 0, 0].size

		e_path = np.empty((pathlen, n), dtype=type_real_np)
		wf_path = np.empty((pathlen, n, n), dtype=type_complex_np)

		for i in range(pathlen):
			e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
			if degenerate_eigenvalues:
				for j in range(int(n/2)):
					wf1 = np.copy(wf_buff[:, 2*j])
					wf2 = np.copy(wf_buff[:, 2*j+1])
					wf_buff[:, 2*j] *= wf2[n-2]
					wf_buff[:, 2*j] -= wf1[n-2]*wf2
					wf_buff[:, 2*j+1] *= wf1[n-1]
					wf_buff[:, 2*j+1] -= wf2[n-1]*wf1
			wf_gauged_entry = np.copy(wf_buff[gidx, :])
			wf_buff[gidx, :] = np.abs(wf_gauged_entry)
			wf_buff[~(np.arange(n) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
			wf_path[i] = wf_buff

		return e_path, wf_path

	@conditional_njit(type_complex_np)
	def dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = h_in_path[:, 0, 0].size

		dx_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		dy_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		if not dm_dynamics_method == 'semiclassics':
			_buf, wf_path = diagonalize_path(h_in_path)
			dwfkx_path, dwfky_path, _buf, _buf = __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
			                                                       hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

			for i in range(pathlen):
				dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
				dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

		return dx_path, dy_path

	return polarization_inter_path_velocity


def make_intraband_current_path_velocity(path, P, sys):
	"""
	    Function that calculates the intraband current from eq. (76 and 77) with or without the
	    anomalous contribution via the Berry curvature
	"""
	E_dir = P.E_dir
	E_ort = P.E_ort

	Nk1 = P.Nk1
	n = P.n
	epsilon = P.epsilon
	type_complex_np = P.type_complex_np
	type_real_np = P.type_real_np
	save_anom = P.save_anom
	Bcurv_path = sys.Bcurv_path
	degenerate_eigenvalues = sys.degenerate_eigenvalues
	gidx = P.gidx

	kx_before_shift = path[:, 0]
	ky_before_shift = path[:, 1]
	pathlen = kx_before_shift.size

	def current_intra_path_velocity(solution, E_field, A_field):

		kx_in_path = kx_before_shift + A_field*E_dir[0]
		ky_in_path = ky_before_shift + A_field*E_dir[1]
		path_after_shift = np.copy(path)
		path_after_shift[:, 0] = kx_in_path
		path_after_shift[:, 1] = ky_in_path  
		pathlen = kx_before_shift.size

		h_in_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp5ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm5ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp5ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm5ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hpexm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hmexm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4expey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp2exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp3exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hp4exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm2exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm3exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
		hm4exmey = np.zeros((pathlen, n, n), dtype=type_complex_np)

		for i in range(n):
			for j in range(n):
				for k in range(pathlen):
					kx = kx_in_path[k]
					ky = ky_in_path[k]
					h_in_path[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky)
					hpex[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky)
					hmex[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky)
					hpey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+P.epsilon)
					hmey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-P.epsilon)
					hp2ex[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky)
					hm2ex[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky)
					hp2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+2*P.epsilon)
					hm2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-2*P.epsilon)
					hp3ex[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky)
					hm3ex[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky)
					hp3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+3*P.epsilon)
					hm3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-3*P.epsilon)
					hp4ex[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky)
					hm4ex[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky)
					hp4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+4*P.epsilon)
					hm4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-4*P.epsilon)
					hp5ex[k, i, j] = sys.hfjit[i][j](kx=kx+5*P.epsilon, ky=ky)
					hm5ex[k, i, j] = sys.hfjit[i][j](kx=kx-5*P.epsilon, ky=ky)
					hp5ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+5*P.epsilon)
					hm5ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-5*P.epsilon)
					hpexpey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+P.epsilon)
					hpexp2ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+2*P.epsilon)
					hpexp3ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+3*P.epsilon)
					hpexp4ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky+4*P.epsilon)
					hpexmey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-P.epsilon)
					hpexm2ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-2*P.epsilon)
					hpexm3ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-3*P.epsilon)
					hpexm4ey[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky-4*P.epsilon)
					hmexpey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+P.epsilon)
					hmexp2ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+2*P.epsilon)
					hmexp3ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+3*P.epsilon)
					hmexp4ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky+4*P.epsilon)
					hmexmey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-P.epsilon)
					hmexm2ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-2*P.epsilon)
					hmexm3ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-3*P.epsilon)
					hmexm4ey[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky-4*P.epsilon)
					hp2expey[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky+P.epsilon)
					hp3expey[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky+P.epsilon)
					hp4expey[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky+P.epsilon)
					hm2expey[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky+P.epsilon)
					hm3expey[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky+P.epsilon)
					hm4expey[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky+P.epsilon)
					hp2exmey[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky-P.epsilon)
					hp3exmey[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky-P.epsilon)
					hp4exmey[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky-P.epsilon)
					hm2exmey[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky-P.epsilon)
					hm3exmey[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky-P.epsilon)
					hm4exmey[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky-P.epsilon)
		J_intra_E_dir = 0
		J_intra_ortho = 0
		J_anom_ortho = np.zeros(n, dtype=type_real_np)

		_buf, _buf, ederivx, ederivy = calculate_ederiv(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
		                                                hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

		ederiv_in_path = E_dir[0] * ederivx + E_dir[1] * ederivy
		ederiv_ortho = E_ort[0] * ederivx + E_ort[1] * ederivy

		Bcurv_path = __berry_curvature(h_in_path, hpex, hmex, hpey, hmey, hp2ex, hm2ex, hp2ey, hm2ey, hp3ex, hm3ex, hp3ey, hm3ey, hp4ex, hm4ex, \
		                               hp4ey, hm4ey, hp5ex, hm5ex, hp5ey, hm5ey, hpexpey, hpexp2ey, hpexp3ey, hpexp4ey, hpexmey, hpexm2ey, hpexm3ey, hpexm4ey, \
		                               hmexpey, hmexp2ey, hmexp3ey, hmexp4ey, hmexmey, hmexm2ey, hmexm3ey, hmexm4ey, hp2expey, hp3expey, hp4expey, hm2expey, \
		                               hm3expey, hm4expey, hp2exmey, hp3exmey, hp4exmey, hm2exmey, hm3exmey, hm4exmey)

		for k in range(Nk1):
			for i in range(n):
				if i < n/2: #A ROUTINE TO SUBSTRACT 1 FOR ALL OCCUPIED BANDS IS NEEDED (or just substract it from each band)
					J_intra_E_dir -= ederiv_in_path[k, i] * (solution[k, i, i].real - 1)
					J_intra_ortho -= ederiv_ortho[k, i] * (solution[k, i, i].real - 1)
				else:
					J_intra_E_dir -= ederiv_in_path[k, i] * solution[k, i, i].real
					J_intra_ortho -= ederiv_ortho[k, i] * solution[k, i, i].real 
				if save_anom:
					J_anom_ortho[i] -= E_field * Bcurv_path[k, i].real * solution[k, i, i].real

		return J_intra_E_dir, J_intra_ortho, J_anom_ortho

	@conditional_njit(type_complex_np)
	def __berry_curvature(h_in_path, hpex, hmex, hpey, hmey, hp2ex, hm2ex, hp2ey, hm2ey, hp3ex, hm3ex, hp3ey, hm3ey, hp4ex, hm4ex, \
	                      hp4ey, hm4ey, hp5ex, hm5ex, hp5ey, hm5ey, hpexpey, hpexp2ey, hpexp3ey, hpexp4ey, hpexmey, hpexm2ey, hpexm3ey, hpexm4ey, \
	                      hmexpey, hmexp2ey, hmexp3ey, hmexp4ey, hmexmey, hmexm2ey, hmexm3ey, hmexm4ey, hp2expey, hp3expey, hp4expey, hm2expey, \
	                      hm3expey, hm4expey, hp2exmey, hp3exmey, hp4exmey, hm2exmey, hm3exmey, hm4exmey):

		pathlen = path[:, 0].size

		dAydx = np.empty((pathlen, n, n), dtype=type_complex_np)
		dAxdy = np.empty((pathlen, n, n), dtype=type_complex_np)

		Ax_plusx, Ay_plusx = dipole_path(hpex, hp2ex, h_in_path, hp3ex, hmex, hp4ex, hm2ex, hp5ex, hm3ex, \
		                                 hpexpey, hpexmey, hpexp2ey, hpexm2ey, hpexp3ey, hpexm3ey, hpexp4ey, hpexm4ey)
		Ax_minusx, Ay_minusx = dipole_path(hmex, h_in_path, hm2ex, hpex, hm3ex, hp2ex, hm4ex, hp3ex, hm5ex, \
		                                   hmexpey, hmexmey, hmexp2ey, hmexm2ey, hmexp3ey, hmexm3ey, hmexp4ey, hmexm4ey)
		Ax_plusy, Ay_plusy = dipole_path(hpey, hpexpey, hmexpey, hp2expey, hm2expey, hp3expey, hm3expey, hp4expey, hm4expey, \
		                                 hp2ey, h_in_path, hp3ey, hmey, hp4ey, hm2ey, hp5ey, hm3ey)
		Ax_minusy, Ay_minusy = dipole_path(hmey, hpexmey, hmexmey, hp2exmey, hm2exmey, hp3exmey, hm3exmey, hp4exmey, hm4exmey, \
		                                   h_in_path, hm2ey, hpey, hm3ey, hp2ey, hm4ey, hp3ey, hm5ey)


		dAxdy = (Ax_plusy - Ax_minusy)/(2*epsilon)
		dAydx = (Ay_plusx - Ay_minusx)/(2*epsilon)

		Bcurv = np.zeros((pathlen, n), dtype=type_complex_np)

		for i in range(n):
			for i_k in range(pathlen):
				Bcurv[i_k, i] = dAxdy[i_k, i, i] - dAydx[i_k, i, i]

		return Bcurv

	@conditional_njit(type_complex_np)
	def dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = h_in_path[:, 0, 0].size

		dx_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
		dy_path = np.zeros((pathlen, n, n), dtype=type_complex_np)

		_buf, wf_path = diagonalize_path(h_in_path)
		dwfkx_path, dwfky_path, _buf, _buf = __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
		                                                       hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

		for i in range(pathlen):
			dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
			dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

		return dx_path, dy_path

	@conditional_njit(type_complex_np)
	def __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = hpex[:, 0, 0].size

		xderivative = np.empty((pathlen, n, n), dtype=type_complex_np)
		yderivative = np.empty((pathlen, n, n), dtype=type_complex_np)

		eplusx, wfplusx = diagonalize_path(hpex)
		eminusx, wfminusx = diagonalize_path(hmex)
		eplusy, wfplusy = diagonalize_path(hpey)
		eminusy, wfminusy = diagonalize_path(hmey)

		eplus2x, wfplus2x = diagonalize_path(hp2ex)
		eminus2x, wfminus2x = diagonalize_path(hm2ex)
		eplus2y, wfplus2y = diagonalize_path(hp2ey)
		eminus2y, wfminus2y = diagonalize_path(hm2ey)

		eplus3x, wfplus3x = diagonalize_path(hp3ex)
		eminus3x, wfminus3x = diagonalize_path(hm3ex)
		eplus3y, wfplus3y = diagonalize_path(hp3ey)
		eminus3y, wfminus3y = diagonalize_path(hm3ey)

		eplus4x, wfplus4x = diagonalize_path(hp4ex)
		eminus4x, wfminus4x = diagonalize_path(hm4ex)
		eplus4y, wfplus4y = diagonalize_path(hp4ey)
		eminus4y, wfminus4y = diagonalize_path(hm4ey)

		xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
		yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
		ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
		ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

		return xderivative, yderivative, ederivx, ederivy


	@conditional_njit(type_complex_np)
	def calculate_ederiv(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

		pathlen = hpex[:, 0, 0].size

		xderivative = np.empty((pathlen, n, n), dtype=type_complex_np)
		yderivative = np.empty((pathlen, n, n), dtype=type_complex_np)

		eplusx, wfplusx = diagonalize_path(hpex)
		eminusx, wfminusx = diagonalize_path(hmex)
		eplusy, wfplusy = diagonalize_path(hpey)
		eminusy, wfminusy = diagonalize_path(hmey)

		eplus2x, wfplus2x = diagonalize_path(hp2ex)
		eminus2x, wfminus2x = diagonalize_path(hm2ex)
		eplus2y, wfplus2y = diagonalize_path(hp2ey)
		eminus2y, wfminus2y = diagonalize_path(hm2ey)

		eplus3x, wfplus3x = diagonalize_path(hp3ex)
		eminus3x, wfminus3x = diagonalize_path(hm3ex)
		eplus3y, wfplus3y = diagonalize_path(hp3ey)
		eminus3y, wfminus3y = diagonalize_path(hm3ey)

		eplus4x, wfplus4x = diagonalize_path(hp4ex)
		eminus4x, wfminus4x = diagonalize_path(hm4ex)
		eplus4y, wfplus4y = diagonalize_path(hp4ey)
		eminus4y, wfminus4y = diagonalize_path(hm4ey)

		xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
		yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
		ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
		ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

		return xderivative, yderivative, ederivx, ederivy


	@conditional_njit(type_complex_np)
	def diagonalize_path(h_in_path):

		pathlen = h_in_path[:, 0, 0].size

		e_path = np.empty((pathlen, n), dtype=type_real_np)
		wf_path = np.empty((pathlen, n, n), dtype=type_complex_np)

		for i in range(pathlen):
			e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
			if degenerate_eigenvalues:
				for j in range(int(n/2)):
					wf1 = np.copy(wf_buff[:, 2*j])
					wf2 = np.copy(wf_buff[:, 2*j+1])
					wf_buff[:, 2*j] *= wf2[n-2]
					wf_buff[:, 2*j] -= wf1[n-2]*wf2
					wf_buff[:, 2*j+1] *= wf1[n-1]
					wf_buff[:, 2*j+1] -= wf2[n-1]*wf1
			wf_gauged_entry = np.copy(wf_buff[gidx, :])
			wf_buff[gidx, :] = np.abs(wf_gauged_entry)
			wf_buff[~(np.arange(n) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
			wf_path[i] = wf_buff

		return e_path, wf_path

	return current_intra_path_velocity
	
