import time
from math import ceil, modf
import numpy as np
from numpy.fft import fftshift, fft, ifftshift, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode
import os

from cued.utility import ConversionFactors as CoFa, ParamsParser
from cued.utility import conditional_njit, evaluate_njit_matrix
from cued.utility import FrequencyContainers, TimeContainers
from cued.utility import MpiHelpers
from cued.plotting import write_and_compile_latex_PDF
from cued.fields import make_electric_field
from cued.kpoint_mesh import hex_mesh, rect_mesh
from cued.observables import *
from cued.rhs_ode import *


def sbe_solver(sys, params):
    """
    Solver for the semiconductor bloch equation ( eq. (39) or (47) in https://arxiv.org/abs/2008.03177)
    for a n band system with numerical calculation of the dipole elements (unfinished - analytical dipoles
    can be used for n=2)

    Author: Adrian Seith (adrian.seith@ur.de)
    Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

    Parameters
    ----------
    sys : class
        Symbolic Hamiltonian of the system
    dipole : class
        Symbolic expression for the dipole elements (eq. (37/38))
    params :
        Parameters from the params.py file
    curvature : class
        Symbolic berry curvature (d(Ax)/d(ky) - d(Ay)/d(kx)) with
        A as the Berry connection (eq. (38))

    Returns
    -------
    params
    ------
    saves parameters of the calculation

    Iexact (file, 8 components)
    ------
    t : np.ndarray
        Nt-dimensional array of time-points
    I_exact_E_dir : np.ndarray
        Nt-dimensional array of current (eq. (59/64)) in E-field direction
    I_exact_ortho : np.ndarray
        Nt-dimensional array of current (eq. (59/64)) orthogonal to E-field
    freq/w : np.ndarray
        Nt-dimensional array of time-points in frequency domain
    Iw_exact_E_dir : np.ndarray
        Nt-dimensional array of fourier trafo of current in E-field direction
    Iw_exact_ortho : np.ndarray
        Nt-dimensional array of fourier trafo of current orthogonal to E-field
    I_exact_E_dir : np.ndarray
        Nt-dimensional array of emission intensity (eq. (51)) in E-field direction
    I_exact_ortho : np.ndarray
        Nt-dimensional array of emission intensity (eq. (51)) orthogonal to E-field

    Iapprox  (file, 8 components)
    -------
    approximate solutions, but same components as Iexact
    """
    # Start time of sbe_solver
    start_time = time.perf_counter()

    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    P = ParamsParser(params)

    P.n = sys.n
    P.n_sheets = 1
    if hasattr(sys, 'n_sheets'):
        P.n_sheets = sys.n_sheets

    # Make Brillouin zone (saved in P)
    make_BZ(P)

    # USER OUTPUT
    ###########################################################################
    if P.user_out:
        print_user_info(P)

    # INITIALIZATIONS
    ###########################################################################

    # Initialize Mpi
    Mpi = MpiHelpers()
    Mpi.local_Nk2_idx_list = Mpi.get_local_idx(P.Nk2)

    # Make containers for time- and frequency- dependent observables
    T = TimeContainers(P)
    W = FrequencyContainers()

    # Make rhs of ode for 2band or nband solver
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
        if P.solver_method in ('bdf', 'adams'):
            solver.set_initial_value(y0, P.t0)\
                .set_f_params(path, sys.dipole_in_path, sys.e_in_path, y0, P.dk)
        elif P.solver_method == 'rk4':
            T.solution_y_vec[:] = y0

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
            if P.solver_method in ('bdf', 'adams'):
                solver.integrate(solver.t + P.dt)
                solver_successful = solver.successful()

            elif P.solver_method == 'rk4':
                T.solution_y_vec = rk_integrate(T.t[ti], T.solution_y_vec, path, sys,
                                                y0, P.dk, P.dt, rhs_ode)

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
    params_name = 'params.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(P.__dict__) + "\n\n")
    paramsfile.write("Runtime: {:.16f} s".format(P.run_time))
    paramsfile.close()

    if P.save_full:
        # write_full_density_mpi(T, P, sys, Mpi)
        S_name = 'Sol_' + P.tail
        np.savez(S_name, t=T.t, solution_full=T.solution_full, paths=P.paths,
                 electric_field=T.electric_field(T.t), A_field=T.A_field)

def make_BZ(P):
        # Form Brillouin Zone
    if P.BZ_type == 'hexagon':
        if P.align == 'K':
            P.E_dir = np.array([1, 0], P.type_real_np)
        elif P.align == 'M':
            P.E_dir = np.array([np.cos(np.radians(-30)),
                                    np.sin(np.radians(-30))], dtype=P.type_real_np)
        P.dk, P.kweight, P.paths = hex_mesh(P)

    elif P.BZ_type == 'rectangle':
        P.E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
                                np.sin(np.radians(P.angle_inc_E_field))], dtype=P.type_real_np)
        P.dk, P.kweight, P.paths = rect_mesh(P)

    P.E_ort = np.array([P.E_dir[1], -P.E_dir[0]])

def make_rhs_ode(P, T, sys):
    if P.solver == '2band':
        if P.n != 2:
            raise AttributeError('2-band solver works for 2-band systems only')
        else:
            rhs_ode = make_rhs_ode_2_band(sys, T.electric_field, P)

    elif P.solver == 'nband':
        rhs_ode = make_rhs_ode_n_band(T.electric_field, P)

    if P.solver_method in ('bdf', 'adams'):
        solver = ode(rhs_ode, jac=None)\
            .set_integrator('zvode', method=P.solver_method, max_step=P.dt)
    elif P.solver_method == 'rk4':
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
        current_exact_path = make_current_exact_path_hderiv(path, P, sys)
        if P.split_current:
            polarization_inter_path = make_polarization_inter_path(P, sys)
            current_intra_path = make_intraband_current_path(path, P, sys)
    else:
        current_exact_path = make_current_exact_bandstructure(path, P, sys)
        if P.split_current:
            polarization_inter_path = make_polarization_inter_bandstructure(P, sys)
            current_intra_path = make_intraband_current_bandstructure(path, P, sys)
    return current_exact_path, polarization_inter_path, current_intra_path


def calculate_solution_at_timestep(solver, Nk2_idx, ti, T, P, Mpi):

    is_first_Nk2_idx = (Mpi.local_Nk2_idx_list[0] == Nk2_idx)

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

    # Only write full density matrix solution if save_full is True
    if P.save_full:
        T.solution_full[:, Nk2_idx, ti, :, :] = T.solution


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
        T.j_anom_ortho[ti] += j_anom_ortho_buf


def rk_integrate(t, y, kpath, sys, y0, dk, dt, rhs_ode):

    k1 = rhs_ode(t,          y,          kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
    k2 = rhs_ode(t + 0.5*dt, y + 0.5*k1, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
    k3 = rhs_ode(t + 0.5*dt, y + 0.5*k2, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)
    k4 = rhs_ode(t +     dt, y +     k3, kpath, sys.dipole_in_path, sys.e_in_path, y0, dk)

    ynew = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return ynew


def initial_condition(P, e_in_path): # Check if this does what it should!
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


def gaussian(t, sigma):
    '''
    Window function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2/sigma**2)

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

        T.j_intra_plus_anom_ortho = T.j_intra_ortho + T.j_anom_ortho


def calculate_fourier(T, P, W):

    # Fourier transforms
    # 1/(3c^3) in atomic units
    prefac_emission = 1/(3*(137.036**3))
    dt_out = T.t[1] - T.t[0]
    ndt_fft = (T.t.size-1)*P.factor_freq_resolution + 1
    W.freq = fftshift(fftfreq(ndt_fft, d=dt_out))

    if P.fourier_window_function == 'gaussian':
         T.window_function = gaussian(T.t, P.gaussian_window_width)
    elif P.fourier_window_function == 'hann':
         T.window_function = hann(T.t)
    elif P.fourier_window_function == 'parzen':
         T.window_function = parzen(T.t)


    W.I_E_dir, W.I_ortho, W.j_E_dir, W.j_ortho =\
        fourier_current_intensity(T.j_E_dir, T.j_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

    # always compute the Fourier transform with hann and parzen window for comparison; this is printed to the latex PDF
    W.I_E_dir_hann, W.I_ortho_hann, W.j_E_dir_hann, W.j_ortho_hann =\
        fourier_current_intensity(T.j_E_dir, T.j_ortho, hann(T.t), dt_out, prefac_emission, W.freq, P)

    W.I_E_dir_parzen, W.I_ortho_parzen, W.j_E_dir_parzen, W.j_ortho_parzen =\
        fourier_current_intensity(T.j_E_dir, T.j_ortho, parzen(T.t), dt_out, prefac_emission, W.freq, P)


    if P.split_current:
        # Approximate current and emission intensity
        W.I_intra_plus_dtP_E_dir, W.I_intra_plus_dtP_ortho, W.j_intra_plus_dtP_E_dir, W.j_intra_plus_dtP_ortho =\
            fourier_current_intensity(T.j_intra_plus_dtP_E_dir, T.j_intra_plus_dtP_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

        # Intraband current and emission intensity
        W.I_intra_E_dir, W.I_intra_ortho, W.j_intra_E_dir, W.j_intra_ortho =\
            fourier_current_intensity(T.j_intra_E_dir, T.j_intra_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

        # Polarization-related current and emission intensity
        W.I_dtP_E_dir, W.I_dtP_ortho, W.dtP_E_dir, W.dtP_ortho =\
            fourier_current_intensity(T.dtP_E_dir, T.dtP_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)

        # Anomalous current, intraband current (de/dk-related) + anomalous current; and emission int.
        W.I_anom_ortho, W.I_intra_plus_anom_ortho, W.j_anom_ortho, W.j_intra_plus_anom_ortho =\
            fourier_current_intensity(T.j_anom_ortho, T.j_intra_plus_anom_ortho, T.window_function, dt_out, prefac_emission, W.freq, P)


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

def write_current_emission_mpi(T, P, W, sys, Mpi):

    # only save data from a single MPI rank
    if Mpi.rank == 0:
        write_current_emission(T, P, W, sys, Mpi)


def write_current_emission(T, P, W, sys, Mpi):

    ##################################################
    # Time data save
    ##################################################
    if P.split_current:
        time_header = ("{:25s}" + " {:27s}"*10)\
            .format("t",
                    "j_E_dir", "j_ortho",
                    "j_intra_E_dir", "j_intra_ortho",
                    "dtP_E_dir", "dtP_ortho",
                    "j_intra_plus_dtP_E_dir", "j_intra_plus_dtP_ortho",
                    "j_anom_ortho", "j_intra_plus_anom_ortho")
        time_output = np.column_stack([T.t.real,
                                       T.j_E_dir.real, T.j_ortho.real,
                                       T.j_intra_E_dir.real, T.j_intra_ortho.real,
                                       T.dtP_E_dir.real, T.dtP_ortho.real,
                                       T.j_intra_plus_dtP_E_dir.real, T.j_intra_plus_dtP_ortho.real,
                                       T.j_anom_ortho.real, T.j_intra_plus_anom_ortho.real])

    elif P.sheet_current:
        print('test')
        time_header =("{:25s}").format('t')
        for i in range(P.n_sheets):
            for j in range(P.n_sheets):
                time_header += (" {:27s}"*2).format(f"j_E_dir[{i},{j}]", f"j_ortho[{i},{j}]")

        time_output = T.t.real
        for i in range(P.n_sheets):
            for j in range(P.n_sheets):
                time_output = np.column_stack((time_output, T.j_E_dir[:, i, j].real, T.j_ortho[:, i, j]) )

    else:
        time_header = ("{:25s}" + " {:27s}"*2)\
            .format("t", "j_E_dir", "j_ortho")
        time_output = np.column_stack([T.t.real,
                                       T.j_E_dir.real, T.j_ortho.real])

    # Make the maximum exponent double digits
    time_output[np.abs(time_output) <= 10e-100] = 0
    time_output[np.abs(time_output) >= 1e+100] = np.inf

    np.savetxt('time_data.dat', time_output, header=time_header, delimiter='   ', fmt="%+.18e")

    ##################################################
    # Frequency data save
    ##################################################
    if P.split_current:
        freq_header = ("{:25s}" + " {:27s}"*30)\
            .format("f/f0",
                    "Re[j_E_dir]", "Im[j_E_dir]", "Re[j_ortho]", "Im[j_ortho]",
                    "I_E_dir", "I_ortho",
                    "Re[j_intra_E_dir]", "Im[j_intra_E_dir]", "Re[j_intra_ortho]", "Im[j_intra_ortho]",
                    "I_intra_E_dir", "I_intra_ortho",
                    "Re[dtP_E_dir]", "Im[dtP_E_dir]", "Re[dtP_ortho]", "Im[dtP_ortho]",
                    "I_dtP_E_dir", "I_dtP_ortho",
                    "Re[j_intra_plus_dtP_E_dir]", "Im[j_intra_plus_dtP_E_dir[", "Re[j_intra_plus_dtP_ortho]", "Im[j_intra_plus_dtP_ortho]",
                    "I_intra_plus_dtP_E_dir", "I_intra_plus_dtP_ortho",
                    "Re[j_anom_ortho]", "Im[j_anom_ortho]", "Re[j_intra_plus_anom_ortho]", "Im[j_intra_plus_anom_ortho]",
                    "I_anom_ortho", "I_intra_plus_anom_ortho")

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
                                       W.j_anom_ortho.real, W.j_anom_ortho.imag, W.j_intra_plus_anom_ortho.real, W.j_intra_plus_anom_ortho.imag,
                                       W.I_anom_ortho.real, W.I_intra_plus_anom_ortho.real])

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

    np.savetxt('frequency_data.dat', freq_output, header=freq_header, delimiter='   ', fmt="%+.18e")

    if P.save_latex_pdf:
        write_and_compile_latex_PDF(T, W, P, sys, Mpi)


def fourier_current_intensity(I_E_dir, I_ortho, window_function, dt_out, prefac_emission, freq, P):

    if P.sheet_current:

        ndt     = np.size(I_E_dir[:, 0, 0])
        ndt_fft = np.size(freq)
        jw_E_dir = np.zeros([ndt_fft, P.n_sheets, P.n_sheets])
        jw_ortho = np.zeros([ndt_fft, P.n_sheets, P.n_sheets])

        for i in range(P.n_sheets):
            for j in range(P.n_sheets):
                I_E_dir_for_fft = np.zeros(ndt_fft)
                I_ortho_for_fft = np.zeros(ndt_fft)

                I_E_dir_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_E_dir[:, i, j]*window_function[:]
                I_ortho_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_ortho[:, i, j]*window_function[:]

                jw_E_dir[:, i, j] = fourier(dt_out, I_E_dir_for_fft)
                jw_ortho[:, i, j] = fourier(dt_out, I_ortho_for_fft)

                I_E_dir[:, i, j] = prefac_emission*(freq**2)*np.abs(jw_E_dir[:, i, j])**2
                I_ortho[:, i, j] = prefac_emission*(freq**2)*np.abs(jw_ortho[:, i, j])**2
   
    else:
        ndt     = np.size(I_E_dir)
        ndt_fft = np.size(freq)

        I_E_dir_for_fft = np.zeros(ndt_fft)
        I_ortho_for_fft = np.zeros(ndt_fft)

        I_E_dir_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_E_dir[:]*window_function[:]
        I_ortho_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_ortho[:]*window_function[:]

        jw_E_dir = fourier(dt_out, I_E_dir_for_fft)
        jw_ortho = fourier(dt_out, I_ortho_for_fft)

        I_E_dir = prefac_emission*(freq**2)*np.abs(jw_E_dir)**2
        I_ortho = prefac_emission*(freq**2)*np.abs(jw_ortho)**2

    return I_E_dir, I_ortho, jw_E_dir, jw_ortho


def print_user_info(P, B0=None, mu=None, incident_angle=None):
    """
    Function that prints the input parameters if usr_info = True
    """

    print("Input parameters:")
    print("Brillouin zone                  = " + P.BZ_type)
    print("Do Semiclassics                 = " + str(P.do_semicl))
    print("ODE solver method               = " + str(P.solver_method))
    print("Precision (default = double)    = " + str(P.precision))
    print("Number of k-points              = " + str(P.Nk))
    print("Order of k-derivative           = " + str(P.dk_order))
    print("Right hand side of ODE          = " + str(P.solver))
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
