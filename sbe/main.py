import time
from math import ceil, modf
import numpy as np
from numpy.fft import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode
import sbe.dipole

from sbe.kpoint_mesh import hex_mesh, rect_mesh
from sbe.utility import ConversionFactors as co
from sbe.utility import conditional_njit
from sbe.utility import parse_params
from sbe.fields import make_electric_field
from sbe.dipole import diagonalize, dipole_elements
from sbe.observables_n_bands import *
from sbe.observables import *
from sbe.rhs_ode import *

def sbe_solver(sys, params, electric_field_function=None):
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
        Int_exact_E_dir : np.ndarray
            Nt-dimensional array of emission intensity (eq. (51)) in E-field direction
        Int_exact_ortho : np.ndarray
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
    P = parse_params(params)

    # USER OUTPUT
    ###########################################################################
    if P.user_out:
        print_user_info(P)
    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if P.BZ_type == 'hexagon':
        _kpnts, paths, area = hex_mesh(P)
        kweight = area/P.Nk
        dk = 1/P.Nk1
        if P.align == 'K':
            E_dir = np.array([1, 0])
        elif P.align == 'M':
            E_dir = np.array([np.cos(np.radians(-30)),
                              np.sin(np.radians(-30))])
        # BZ_plot(_kpnts, paths, P)
    elif P.BZ_type == 'rectangle':
        E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
                          np.sin(np.radians(P.angle_inc_E_field))])
        dk, kweight, _kpnts, paths = rect_mesh(P, E_dir, P.type_real_np)
        # BZ_plot(_kpnts, a, b1, b2, paths)

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # Calculate the systems properties (hamiltonian, eigensystem, dipoles, berry curvature)
    if P.system == 'ana':
        h_sym, ef_sym, wf_sym, _ediff_sym = sys.eigensystem(gidx=P.gidx)
        dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        curvature = sbe.dipole.SymbolicCurvature(h_sym, dipole.Ax, dipole.Ay)
        P.n = 2

    if P.system == 'num':
        hnp = sys.numpy_hamiltonian()  
        P.n = np.size(hnp(kx=0, ky=0)[:, 0])
        dipole_x, dipole_y = dipole_elements(P, hnp, paths)
        e, wf = diagonalize(P, hnp, paths)
        curvature = 0   

    # Initialize electric_field, create rhs of ode and initialize solver

    if electric_field_function is None:
        electric_field = make_electric_field(P.E0, P.w, P.alpha, P.chirp, P.phase, \
            P.type_real_np)
    else:
        electric_field = electric_field_function
    
    # Make rhs of ode for 2band or nband solver
    if P.solver == '2band':
        if P.n != 2: 
            raise AttributeError('2-band solver works for 2-band systems only')
        if P.system == 'ana':
           rhs_ode = make_rhs_ode_2_band(sys, dipole, E_dir, electric_field, P)
        elif P.system == 'num':
            if P.gauge == 'length':
                rhs_ode = make_rhs_ode_2_band(0, 0, E_dir, electric_field, P)
            if P.gauge == 'velocity':
                raise AttributeError('numerical evaluation of the system not compatible with velocity gauge')
    elif P.solver == 'nband':
        rhs_ode = make_rhs_ode_n_band(E_dir, electric_field, P)

    if P.solver_method in ('bdf', 'adams'):    
        solver = ode(rhs_ode, jac=None)\
            .set_integrator('zvode', method=P.solver_method, max_step=P.dt)

    # Make containers used in solver
    t, A_field, E_field, solution, solution_y_vec, J_exact_E_dir, J_exact_ortho, \
        J_intra_E_dir, J_intra_ortho, P_inter_E_dir, P_inter_ortho, J_anom_ortho = solution_container(P)

    dipole_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    dipole_ortho = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    e_in_path = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)  

    # Only define full density matrix solution if save_full is True
    if P.save_full:
        solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, P.n, P.n), dtype=P.type_complex_np)
    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone

    for Nk2_idx, path in enumerate(paths):

        # parallelization if requested in runscript
        if P.Nk2_idx_ext != Nk2_idx and P.Nk2_idx_ext >= 0: 
            continue

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Evaluate the dipole components along the path
        if P.system == 'num':    
            if P.do_semicl:
                0
            # Calculate the dot products E_dir.d_nm(k).
            # To be multiplied by E-field magnitude later.            
            else:
                dipole_in_path = (E_dir[0]*dipole_x[:, Nk2_idx, :, :] + \
                    E_dir[1]*dipole_y[:, Nk2_idx, :, :])
                dipole_ortho = (E_ort[0]*dipole_x[:, Nk2_idx, :, :] + \
                    E_ort[1]*dipole_y[:, Nk2_idx, :, :])
                e_in_path = e[:, Nk2_idx, :]
                wf_in_path = wf[:, Nk2_idx, :, :]   #not in E-dir!

        elif P.system == 'ana':
            if P.do_semicl:
                0
            else:
                # Calculate the dipole components along the path
                di_00x = dipole.Axfjit[0][0](kx=kx_in_path, ky=ky_in_path)
                di_01x = dipole.Axfjit[0][1](kx=kx_in_path, ky=ky_in_path)
                di_11x = dipole.Axfjit[1][1](kx=kx_in_path, ky=ky_in_path)
                di_00y = dipole.Ayfjit[0][0](kx=kx_in_path, ky=ky_in_path)
                di_01y = dipole.Ayfjit[0][1](kx=kx_in_path, ky=ky_in_path)
                di_11y = dipole.Ayfjit[1][1](kx=kx_in_path, ky=ky_in_path)

                # Calculate the dot products E_dir.d_nm(k).
                # To be multiplied by E-field magnitude later.
                dipole_in_path[:, 0, 1] = E_dir[0]*di_01x + E_dir[1]*di_01y
                dipole_in_path[:, 1, 0] = dipole_in_path[:, 0, 1].conjugate()
                dipole_in_path[:, 0, 0] = E_dir[0]*di_00x + E_dir[1]*di_00y
                dipole_in_path[:, 1, 1] = E_dir[0]*di_11x + E_dir[1]*di_11y

                dipole_ortho[:, 0, 1] = E_ort[0]*di_01x + E_ort[1]*di_01y
                dipole_ortho[:, 1, 0] = dipole_ortho[:, 0, 1].conjugate()
                dipole_ortho[:, 0, 0] = E_ort[0]*di_00x + E_ort[1]*di_00y
                dipole_ortho[:, 1, 1] = E_ort[0]*di_11x + E_ort[1]*di_11y

            e_in_path[:, 0] = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
            e_in_path[:, 1] = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)

            ecv_in_path = e_in_path[:, 1] - e_in_path[:, 0]

            wf_in_path = np.empty([P.Nk1, 2, 2], dtype=P.type_complex_np)   #not in E-dir!
            Ujit = sys.Ujit
            wf_in_path[:, 0, 0] = Ujit[0][0](kx=kx_in_path, ky=ky_in_path)
            wf_in_path[:, 0, 1] = Ujit[0][1](kx=kx_in_path, ky=ky_in_path)
            wf_in_path[:, 1, 0] = Ujit[1][0](kx=kx_in_path, ky=ky_in_path)
            wf_in_path[:, 1, 1] = Ujit[1][1](kx=kx_in_path, ky=ky_in_path)

        # Prepare calculations of observables
        if P.system == 'ana':
            if P.gauge == 'length':
                current_exact_path = make_emission_exact_path_length(sys, path, E_dir, curvature, P)
            if P.gauge == 'velocity':
                current_exact_path = make_emission_exact_path_velocity(sys, path, E_dir, curvature, P)
            if P.save_approx:
                polarization_inter_path = make_polarization_path(dipole, path, E_dir, P)
                current_intra_path = make_current_path(sys, path, E_dir, curvature, P)

        if P.system == 'num':
            current_exact_path = make_current_exact_path_hderiv(P, hnp, paths, wf_in_path, E_dir, Nk2_idx)
            if P.save_approx:
                polarization_inter_path = make_polarization_inter_path(P, dipole_in_path, dipole_ortho)
                current_intra_path = make_intraband_current_path(P, hnp, E_dir, paths, Nk2_idx)


        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = initial_condition(P, e_in_path)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        if P.solver_method in ('bdf', 'adams'):
            solver.set_initial_value(y0, P.t0)\
            .set_f_params(path, dipole_in_path, e_in_path, y0, dk)
        elif P.solver_method == 'rk4':
            solution_y_vec[:] = y0

        # Propagate through time
        # Index of current integration time step
        ti = 0
        solver_successful = True

        while solver_successful and ti < P.Nt:
            # User output of integration progress
            if (ti % (P.Nt//20) == 0 and P.user_out):
                print('{:5.2f}%'.format((ti/P.Nt)*100))

            if P.solver_method in ('bdf', 'adams'):
                # Do not append the last element (A_field)
                solution = solver.y[:-1].reshape(P.Nk1, P.n, P.n)

                # Construct time array only once
                if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
                    # Construct time and A_field only in first round
                    t[ti] = solver.t
                    A_field[ti] = solver.y[-1].real
                    E_field[ti] = electric_field(t[ti])

            elif P.solver_method == 'rk4':
                # Do not append the last element (A_field)
                solution = solution_y_vec[:-1].reshape(P.Nk1, P.n, P.n)

                # Construct time array only once
                if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
                    # Construct time and A_field only in first round
                    t[ti] = ti*P.dt + P.t0
                    A_field[ti] = solution_y_vec[-1].real
                    E_field[ti] = electric_field(t[ti])
       
            # Only write full density matrix solution if save_full is True
            if P.save_full:
                solution_full[:, Nk2_idx, ti, :, :] = solution

            # Calculate the currents at the timestep ti
            if P.system == 'ana':
                J_exact_E_dir_buf, J_exact_ortho_buf = current_exact_path(solution.reshape(P.Nk1, 4), E_field[ti], A_field[ti])
            elif P.system == 'num':
                J_exact_E_dir_buf, J_exact_ortho_buf = current_exact_path(solution)
            J_exact_E_dir[ti] += J_exact_E_dir_buf
            J_exact_ortho[ti] += J_exact_ortho_buf

            if P.save_approx:
                if P.system == 'ana':
                    P_inter_E_dir_buf, P_inter_ortho_buf = polarization_inter_path(solution[:, 1, 0], A_field[ti])
                    J_intra_E_dir_buf, J_intra_ortho_buf, J_anom_ortho_buf = current_intra_path(solution[:,0,0], solution[:, 1, 1], A_field[ti], E_field[ti])
                elif P.system == 'num':
                    P_inter_E_dir_buf, P_inter_ortho_buf = polarization_inter_path(solution)
                    J_intra_E_dir_buf, J_intra_ortho_buf, J_anom_ortho_buf = current_intra_path(solution)

                P_inter_E_dir[ti] += P_inter_E_dir_buf
                P_inter_ortho[ti] += P_inter_ortho_buf
                J_intra_E_dir[ti] += J_intra_E_dir_buf
                J_intra_ortho[ti] += J_intra_ortho_buf
                J_anom_ortho[ti] += J_anom_ortho_buf
            
            # Integrate one integration time step
            if P.solver_method in ('bdf', 'adams'):
                solver.integrate(solver.t + P.dt)
                solver_successful = solver.successful()

            elif P.solver_method == 'rk4':
                solution_y_vec = rk_integrate(t[ti], solution_y_vec, path, dipole_in_path, e_in_path, \
                                              y0, dk, P.dt, rhs_ode)

            # Increment time counter
            ti += 1

    # End time of solver loop
    end_time = time.perf_counter()

    # Write solutions
    # Filename tail
    tail = 'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}_solver_{:s}_dk_order{}'\
        .format(P.E0_MVpcm, P.w_THz, P.alpha_fs, P.gauge, P.t0_fs, P.dt_fs, P.Nk1, P.Nk2, P.T1_fs, P.T2_fs, P.chirp_THz, P.phase, P.solver_method, P.dk_order)

    write_current_emission(tail, kweight, t, J_exact_E_dir, J_exact_ortho,
                           J_intra_E_dir, J_intra_ortho, P_inter_E_dir, P_inter_ortho, J_anom_ortho, P)


    # Save the parameters of the calculation
    run_time = end_time - start_time
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(P.__dict__) + "\n\n")
    paramsfile.write("Runtime: {:.16f} s".format(run_time))
    paramsfile.close()

    if P.save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution_full=solution_full, paths=paths,
                 electric_field=electric_field(t), A_field=A_field)


def rk_integrate(t, y, kpath, dipole_in_path, e_in_path, y0, dk, \
                 dt, rhs_ode):

    k1 = rhs_ode(t,          y,          kpath, dipole_in_path, e_in_path, y0, dk)
    k2 = rhs_ode(t + 0.5*dt, y + 0.5*k1, kpath, dipole_in_path, e_in_path, y0, dk)
    k3 = rhs_ode(t + 0.5*dt, y + 0.5*k2, kpath, dipole_in_path, e_in_path, y0, dk)
    k4 = rhs_ode(t +     dt, y +     k3, kpath, dipole_in_path, e_in_path, y0, dk)

    ynew = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return ynew

def solution_container(P, zeeman=False):
    """
        Function that builds the containers on which the solutions of the SBE,
        as well as the currents will be written
    """
    # Solution containers
    t = np.zeros(P.Nt, dtype=P.type_real_np)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    solution = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_complex_np)

    # For hand-made Runge-Kutta method, we need the solution as array with
    # a single index
    solution_y_vec = np.zeros((((P.n)**2)*(P.Nk1)+1), dtype=P.type_complex_np)

    A_field = np.zeros(P.Nt, dtype=P.type_real_np)
    E_field = np.zeros(P.Nt, dtype=P.type_real_np)

    J_exact_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
    J_exact_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

    if P.save_approx:
        J_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        J_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
        P_inter_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        P_inter_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
        J_anom_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

    else:
        J_E_dir = None
        J_ortho = None
        P_inter_E_dir = None
        P_inter_ortho = None
        J_anom_ortho = None

    if zeeman:
        Zee_field = np.zeros((P.Nt, 3), dtype=P.type_real_np)
        return t, A_field, E_field, solution, J_exact_E_dir, J_exact_ortho, J_E_dir, J_ortho, \
            P_inter_E_dir, P_inter_ortho, Zee_field

    return t, A_field, E_field, solution, solution_y_vec, J_exact_E_dir, J_exact_ortho, \
        J_E_dir, J_ortho, P_inter_E_dir, P_inter_ortho, J_anom_ortho


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

#    dx = np.gradient(x)
#    dy = np.gradient(y)
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

def gaussian(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    # sigma = sqrt(2)*alpha
    # # 1/(2*np.sqrt(np.pi)*alpha)*np.exp(-t**2/(2*alpha)**2)
    return np.exp(-t**2/(2*alpha)**2)


def write_current_emission(tail, kweight, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho, J_anom_ortho, P):
    """
        Calculates the Emission Intensity I(omega) (eq. 51 in https://arxiv.org/abs/2008.03177)

        Author:
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters
        ----------

        tail : str
        kweight : float
        w : float
            driving pulse frequency
        t : ndarray
            array of the time points corresponding to a solution of the sbe
        I_exact_E_dir: ndarray
            exact emission j(t) in E-field direction
        I_exact_ortho : ndarray
            exact emission j(t) orthogonal to E-field
        J_E_dir : ndarray
            approximate emission j(t) in E-field direction
        J_E_ortho : ndarray
            approximate emission j(t) orthogonal to E-field
        P_E_dir : ndarray
            polarization E-field direction
        P_E_ortho : ndarray
            polarization orthogonal to E-field
        gaussian_envelope : function
            gaussian function to multiply to a function before Fourier transform
        save_approx : boolean
            determines whether approximate solutions should be saved
        save_txt : boolean
            determines whether a .txt file with the soluion should be saved

        Returns:
        --------

        savefiles (see documentation of sbe_solver())
    """
    # Fourier transforms
    # 1/(3c^3) in atomic units
    prefac_emission = 1/(3*(137.036**3))
    dt_out = t[1] - t[0]
    ndt_fft = (t.size-1)*P.factor_freq_resolution + 1
    freq = fftshift(fftfreq(ndt_fft, d=dt_out))
    gaussian_envelope = gaussian(t, P.alpha)

    if P.save_approx:
        I_intra_E_dir = J_E_dir*kweight
        I_intra_ortho = J_ortho*kweight

        I_inter_E_dir = diff(t, P_E_dir)*kweight
        I_inter_ortho = diff(t, P_ortho)*kweight

        I_anom_ortho  = J_anom_ortho*kweight

        # Eq. (81( SBE formalism paper
        I_deph_E_dir = 1/P.T2*P_E_dir*kweight
        I_deph_ortho = 1/P.T2*P_ortho*kweight

        I_E_dir = I_intra_E_dir + I_inter_E_dir
        I_ortho = I_intra_ortho + I_inter_ortho

        I_intra_plus_anom_ortho = I_intra_ortho + I_anom_ortho

        I_without_deph_E_dir = I_exact_E_dir - I_deph_E_dir
        I_without_deph_ortho = I_exact_ortho - I_deph_ortho

        I_intra_plus_deph_E_dir = I_intra_E_dir + I_deph_E_dir
        I_intra_plus_deph_ortho = I_intra_ortho + I_deph_ortho

        # Approximate current and emission intensity
        Int_E_dir, Int_ortho, Iw_E_dir, Iw_ortho = fourier_current_intensity(
             I_E_dir, I_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        # Intraband current and emission intensity
        Int_intra_E_dir, Int_intra_ortho, Iw_intra_E_dir, Iw_intra_ortho = fourier_current_intensity(
             I_intra_E_dir, I_intra_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        # Polarization-related current and emission intensity
        Int_inter_E_dir, Int_inter_ortho, Iw_inter_E_dir, Iw_inter_ortho = fourier_current_intensity(
             I_inter_E_dir, I_inter_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        # Anomalous current, intraband current (de/dk-related) + anomalous current; and emission int.
        Int_anom_ortho, Int_intra_plus_anom_ortho, Iw_anom_ortho, Iw_intra_plus_anom_ortho = \
             fourier_current_intensity( I_anom_ortho, I_intra_plus_anom_ortho,
                                        gaussian_envelope, dt_out, prefac_emission, freq)

        # Total current without dephasing current and respectice emission intensity
        Int_without_deph_E_dir, Int_without_deph_ortho, Iw_without_deph_E_dir, Iw_without_deph_ortho = fourier_current_intensity(
             I_without_deph_E_dir, I_without_deph_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        # Total current without dephasing current and respectice emission intensity
        Int_intra_plus_deph_E_dir, Int_intra_plus_deph_ortho, Iw_intra_plus_deph_E_dir, Iw_intra_plus_deph_ortho = fourier_current_intensity(
             I_intra_plus_deph_E_dir, I_intra_plus_deph_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        I_approx_name = 'Iapprox_' + tail

        np.save(I_approx_name, [t, I_E_dir, I_ortho,
                                freq/P.w, Iw_E_dir, Iw_ortho,
                                Int_E_dir, Int_ortho,
                                I_intra_E_dir, I_intra_ortho,
                                Int_intra_E_dir, Int_intra_ortho,
                                I_inter_E_dir, I_inter_ortho,
                                Int_inter_E_dir, Int_inter_ortho,
                                I_anom_ortho, I_intra_plus_anom_ortho,
                                Int_anom_ortho, Int_intra_plus_anom_ortho,
                                Int_without_deph_E_dir, Int_without_deph_ortho,
                                Int_intra_plus_deph_E_dir, Int_intra_plus_deph_ortho] )

        if P.save_txt:
            np.savetxt(I_approx_name + '.dat',
                       np.column_stack([t.real, I_E_dir.real, I_ortho.real,
                                        (freq/P.w).real, Iw_E_dir.real, Iw_E_dir.imag,
                                        Iw_ortho.real, Iw_ortho.imag,
                                        Int_E_dir.real, Int_ortho.real]),
                       header="t, I_E_dir, I_ortho, freqw/w, Re(Iw_E_dir), Im(Iw_E_dir), Re(Iw_ortho), Im(Iw_ortho), Int_E_dir, Int_ortho",
                       fmt='%+.18e')

    ##############################################################
    # Conditional save of exact formula
    ##############################################################
    # kweight is different for rectangle and hexagon
    if P.save_exact:
        I_exact_E_dir *= kweight
        I_exact_ortho *= kweight

        Int_exact_E_dir, Int_exact_ortho, Iw_exact_E_dir, Iw_exact_ortho = fourier_current_intensity(
                I_exact_E_dir, I_exact_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        I_exact_name = 'Iexact_' + tail
        np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho,
                            freq/P.w, Iw_exact_E_dir, Iw_exact_ortho,
                            Int_exact_E_dir, Int_exact_ortho])

        if P.save_txt and P.factor_freq_resolution == 1:
            np.savetxt(I_exact_name + '.dat',
                    np.column_stack([t.real, I_exact_E_dir.real, I_exact_ortho.real,
                                        (freq/P.w).real, Iw_exact_E_dir.real, Iw_exact_E_dir.imag,
                                        Iw_exact_ortho.real, Iw_exact_ortho.imag,
                                        Int_exact_E_dir.real, Int_exact_ortho.real]),
                    header="t, I_exact_E_dir, I_exact_ortho, freqw/w, Re(Iw_exact_E_dir), Im(Iw_exact_E_dir), Re(Iw_exact_ortho), Im(Iw_exact_ortho), Int_exact_E_dir, Int_exact_ortho",
                    fmt='%+.18e')


def fourier_current_intensity(I_E_dir, I_ortho, gaussian_envelope, dt_out, prefac_emission, freq):

    ndt     = np.size(I_E_dir)
    ndt_fft = np.size(freq)

    I_E_dir_for_fft = np.zeros(ndt_fft)
    I_ortho_for_fft = np.zeros(ndt_fft)

    I_E_dir_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_E_dir[:]*gaussian_envelope[:]
    I_ortho_for_fft[ (ndt_fft-ndt)//2 : (ndt_fft+ndt)//2 ] = I_ortho[:]*gaussian_envelope[:]

    Iw_E_dir = fourier(dt_out, I_E_dir_for_fft)
    Iw_ortho = fourier(dt_out, I_ortho_for_fft)

    Int_E_dir = prefac_emission*(freq**2)*np.abs(Iw_E_dir)**2
    Int_ortho = prefac_emission*(freq**2)*np.abs(Iw_ortho)**2

    return Int_E_dir, Int_ortho, Iw_E_dir, Iw_ortho


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
    print("Eigensystem and dipoles         = " + str(P.system))
    print("Right hand side of ODE          = " + str(P.solver))
    if P.BZ_type == 'hexagon':
        print("Driving field alignment         = " + P.align)
    elif P.BZ_type == 'rectangle':
        print("Driving field direction         = " + str(P.angle_inc_E_field))
    if B0 is not None:
        print("Incident angle                  = " + str(np.rad2deg(incident_angle)))
    print("Driving amplitude (MV/cm)[a.u.] = " + "("
          + '{:.6f}'.format(P.E0_MVpcm) + ")"
          + "[" + '{:.6f}'.format(P.E0) + "]")
    if B0 is not None:
        print("Magnetic amplitude (T)[a.u.]    = " + "("
              + '%.6f'%(B0*co.au_to_T) + ")"
              + "[" + '%.6f'%(B0) + "]")
        print("Magnetic moments ", mu)
    print("Pulse Frequency (THz)[a.u.]     = " + "("
          + '{:.6f}'.format(P.w_THz) + ")"
          + "[" + '{:.6f}'.format(P.w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "("
          + '{:.6f}'.format(P.alpha_fs) + ")"
          + "[" + '{:.6f}'.format(P.alpha) + "]")
    print("Chirp rate (THz)[a.u.]          = " + "("
          + '{:.6f}'.format(P.chirp_THz) + ")"
          + "[" + '{:.6f}'.format(P.chirp) + "]")
    print("Damping time (fs)[a.u.]         = " + "("
          + '{:.6f}'.format(P.T2_fs) + ")"
          + "[" + '{:.6f}'.format(P.T2) + "]")
    print("Total time (fs)[a.u.]           = " + "("
          + '{:.6f}'.format(P.tf_fs - P.t0_fs) + ")"
          + "[" + '{:.5f}'.format(P.tf - P.t0) + "]")
    print("Time step (fs)[a.u.]            = " + "("
          + '{:.6f}'.format(P.dt_fs) + ")"
          + "[" + '{:.6f}'.format(P.dt) + "]")


def BZ_plot(kpnts, a, b1, b2, paths, si_units=True):
    """
        Function that plots the brillouin zone
    """
    if si_units:
        a *= co.au_to_as
        kpnts *= co.as_to_au
        b1 *= co.as_to_au
        b2 *= co.as_to_au

    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = plt.figure(figsize=(10, 10))
    ax = BZ_fig.add_subplot(111, aspect='equal')

    for b in ((0, 0), b1, -b1, b2, -b2, b1+b2, -b1-b2):
        poly = RegularPolygon(b, 6, radius=R, orientation=np.pi/6, fill=False)
        ax.add_patch(poly)

#    ax.arrow(-0.5*E_dir[0], -0.5*E_dir[1], E_dir[0], E_dir[1],
#             width=0.005, alpha=0.5, label='E-field')

    plt.scatter(0, 0, s=15, c='black')
    plt.text(0.01, 0.01, r'$\Gamma$')
    plt.scatter(r*np.cos(-np.pi/6), r*np.sin(-np.pi/6), s=15, c='black')
    plt.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, r'$M$')
    plt.scatter(R, 0, s=15, c='black')
    plt.text(R, 0.02, r'$K$')
    plt.scatter(kpnts[:, 0], kpnts[:, 1], s=10)
    plt.xlim(-7.0/a, 7.0/a)
    plt.ylim(-7.0/a, 7.0/a)

    if si_units:
        plt.xlabel(r'$k_x \text{ in } 1/\si{\angstrom}$')
        plt.ylabel(r'$k_y \text{ in } 1/\si{\angstrom}$')
    else:
        plt.xlabel(r'$k_x \text{ in } 1/a_0$')
        plt.ylabel(r'$k_y \text{ in } 1/a_0$')

    for path in paths:
        if si_units:
            plt.plot(co.as_to_au*path[:, 0], co.as_to_au*path[:, 1])
        else:
            plt.plot(path[:, 0], path[:, 1])

    plt.show()
