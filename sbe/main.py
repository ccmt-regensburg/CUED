import time
from math import ceil, modf
import numpy as np
from numpy.fft import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode
import sbe.dipole

from sbe.utility import ConversionFactors as co
from sbe.utility import conditional_njit
from sbe.utility import parse_params, time_containers, system_properties
from sbe.utility import write_and_compile_latex_PDF
from sbe.fields import make_electric_field
from sbe.dipole import diagonalize, dipole_elements
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

    # Calculate the systems properties (hamiltonian, eigensystem, dipoles, berry curvature, BZ, electric field)

    S = system_properties(P, sys)
    
    # Make containers used in solver

    T = time_containers(P, electric_field_function)

    # Make rhs of ode for 2band or nband solver
    rhs_ode, solver = make_rhs_ode(P, S, T)

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone

    for Nk2_idx, path in enumerate(S.paths):

        if P.user_out:
            print('Solving SBE for Path', Nk2_idx+1)

        # parallelization if requested in runscript
        if P.Nk2_idx_ext != Nk2_idx and P.Nk2_idx_ext >= 0: 
            continue

        # Evaluate the dipole components along the path
        
        calculate_system_in_path(path, Nk2_idx, P, S)

        # Prepare calculations of observables

        current_exact_path, polarization_inter_path, current_intra_path = prepare_current_calculations(path, Nk2_idx, S, P)

        # Initialize the values of of each k point vector

        y0 = initial_condition(P, S.e_in_path)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        if P.solver_method in ('bdf', 'adams'):
            solver.set_initial_value(y0, P.t0)\
            .set_f_params(path, S.dipole_in_path, S.e_in_path, y0, S.dk)
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

            calculate_solution_at_timestep(solver, Nk2_idx, ti, T, P)

            # Calculate the currents at the timestep ti

            calculate_currents(ti, current_exact_path, polarization_inter_path, current_intra_path, T, P)

            # Integrate one integration time step
            if P.solver_method in ('bdf', 'adams'):
                solver.integrate(solver.t + P.dt)
                solver_successful = solver.successful()

            elif P.solver_method == 'rk4':
                T.solution_y_vec = rk_integrate(T.t[ti], T.solution_y_vec, path, S, \
                                              y0, S.dk, P.dt, rhs_ode)

            # Increment time counter
            ti += 1

    # End time of solver loop
    end_time = time.perf_counter()
    S.run_time = end_time - start_time

    # Write solutions

    write_current_emission(S, T, P)

    # Save the parameters of the calculation
    params_name = 'params_' + P.tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(P.__dict__) + "\n\n")
    paramsfile.write("Runtime: {:.16f} s".format(S.run_time))
    paramsfile.close()

    if P.save_full:
        S_name = 'Sol_' + P.tail
        np.savez(S_name, t=T.t, solution_full=T.solution_full, paths=S.paths,
                 electric_field=T.electric_field(T.t), A_field=T.A_field)

def make_rhs_ode(P, S, T):
    if P.solver == '2band':
        if P.n != 2: 
            raise AttributeError('2-band solver works for 2-band systems only')
        if P.system == 'ana':
           rhs_ode = make_rhs_ode_2_band(S.sys, S.dipole, S.E_dir, T.electric_field, P)
        elif P.system == 'num':
            if P.gauge == 'length':
                rhs_ode = make_rhs_ode_2_band(0, 0, S.E_dir, T.electric_field, P)
            if P.gauge == 'velocity':
                raise AttributeError('numerical evaluation of the system not compatible with velocity gauge')
    elif P.solver == 'nband':
        rhs_ode = make_rhs_ode_n_band(S.E_dir, T.electric_field, P)

    if P.solver_method in ('bdf', 'adams'):    
        solver = ode(rhs_ode, jac=None)\
            .set_integrator('zvode', method=P.solver_method, max_step=P.dt)
    elif P.solver_method == 'rk4':
        solver = 0
        
    return rhs_ode, solver

def calculate_system_in_path(path, Nk2_idx, P, S):
    
    sys = S.sys

    # Retrieve the set of k-points for the current path
    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]

    if P.system == 'num':    
        if P.do_semicl:
            0
        # Calculate the dot products E_dir.d_nm(k).
        # To be multiplied by E-field magnitude later.            
        else:
            S.dipole_in_path = (S.E_dir[0]*S.dipole_x[:, Nk2_idx, :, :] + \
                S.E_dir[1]*S.dipole_y[:, Nk2_idx, :, :])
            S.dipole_ortho = (S.E_ort[0]*S.dipole_x[:, Nk2_idx, :, :] + \
                S.E_ort[1]*S.dipole_y[:, Nk2_idx, :, :])
            S.e_in_path = S.e[:, Nk2_idx, :]
            S.wf_in_path = S.wf[:, Nk2_idx, :, :]   #not in E-dir!

    elif P.system == 'ana':
        if P.do_semicl:
            0
        else:
            # Calculate the dipole components along the path
            di_00x = S.dipole.Axfjit[0][0](kx=kx_in_path, ky=ky_in_path)
            di_01x = S.dipole.Axfjit[0][1](kx=kx_in_path, ky=ky_in_path)
            di_11x = S.dipole.Axfjit[1][1](kx=kx_in_path, ky=ky_in_path)
            di_00y = S.dipole.Ayfjit[0][0](kx=kx_in_path, ky=ky_in_path)
            di_01y = S.dipole.Ayfjit[0][1](kx=kx_in_path, ky=ky_in_path)
            di_11y = S.dipole.Ayfjit[1][1](kx=kx_in_path, ky=ky_in_path)

            # Calculate the dot products E_dir.d_nm(k).
            # To be multiplied by E-field magnitude later.
            S.dipole_in_path[:, 0, 1] = S.E_dir[0]*di_01x + S.E_dir[1]*di_01y
            S.dipole_in_path[:, 1, 0] = S.dipole_in_path[:, 0, 1].conjugate()
            S.dipole_in_path[:, 0, 0] = S.E_dir[0]*di_00x + S.E_dir[1]*di_00y
            S.dipole_in_path[:, 1, 1] = S.E_dir[0]*di_11x + S.E_dir[1]*di_11y

            S.dipole_ortho[:, 0, 1] = S.E_ort[0]*di_01x + S.E_ort[1]*di_01y
            S.dipole_ortho[:, 1, 0] = S.dipole_ortho[:, 0, 1].conjugate()
            S.dipole_ortho[:, 0, 0] = S.E_ort[0]*di_00x + S.E_ort[1]*di_00y
            S.dipole_ortho[:, 1, 1] = S.E_ort[0]*di_11x + S.E_ort[1]*di_11y

        S.e_in_path[:, 0] = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
        S.e_in_path[:, 1] = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)

        Ujit = sys.Ujit
        S.wf_in_path[:, 0, 0] = Ujit[0][0](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 0, 1] = Ujit[0][1](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 1, 0] = Ujit[1][0](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 1, 1] = Ujit[1][1](kx=kx_in_path, ky=ky_in_path)

def prepare_current_calculations(path, Nk2_idx, S, P):

    if P.system == 'ana':
        if P.gauge == 'length':
            current_exact_path = make_emission_exact_path_length(path, S, P)
        if P.gauge == 'velocity':
            current_exact_path = make_emission_exact_path_velocity(path, S, P)
        if P.save_approx:
            polarization_inter_path = make_polarization_path(path, S, P)
            current_intra_path = make_current_path(path, S, P)

    if P.system == 'num':
        current_exact_path = make_current_exact_path_hderiv(Nk2_idx, S, P)
        if P.save_approx:
            polarization_inter_path = make_polarization_inter_path(S, P)
            current_intra_path = make_intraband_current_path(Nk2_idx, S, P)

    return current_exact_path, polarization_inter_path, current_intra_path
        
def calculate_solution_at_timestep(solver, Nk2_idx, ti, T, P):

    if P.solver_method in ('bdf', 'adams'):
        # Do not append the last element (A_field)
        T.solution = solver.y[:-1].reshape(P.Nk1, P.n, P.n)

        # Construct time array only once
        if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
            # Construct time and A_field only in first round
            T.t[ti] = solver.t
            T.A_field[ti] = solver.y[-1].real
            T.E_field[ti] = T.electric_field(T.t[ti])

    elif P.solver_method == 'rk4':
        # Do not append the last element (A_field)
        T.solution = T.solution_y_vec[:-1].reshape(P.Nk1, P.n, P.n)

        # Construct time array only once
        if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
            # Construct time and A_field only in first round
            T.t[ti] = ti*P.dt + P.t0
            T.A_field[ti] = T.solution_y_vec[-1].real
            T.E_field[ti] = T.electric_field(T.t[ti])

    # Only write full density matrix solution if save_full is True
    if P.save_full:
        T.solution_full[:, Nk2_idx, ti, :, :] = T.solution

def calculate_currents(ti, current_exact_path, polarization_inter_path, current_intra_path, T, P):
    if P.system == 'ana':
        J_exact_E_dir_buf, J_exact_ortho_buf = current_exact_path(T.solution.reshape(P.Nk1, 4), T.E_field[ti], T.A_field[ti])
    elif P.system == 'num':
        J_exact_E_dir_buf, J_exact_ortho_buf = current_exact_path(T.solution)
    T.J_exact_E_dir[ti] += J_exact_E_dir_buf
    T.J_exact_ortho[ti] += J_exact_ortho_buf

    if P.save_approx:
        if P.system == 'ana':
            P_inter_E_dir_buf, P_inter_ortho_buf = polarization_inter_path(T.solution[:, 1, 0], T.A_field[ti])
            J_intra_E_dir_buf, J_intra_ortho_buf, J_anom_ortho_buf = current_intra_path(T.solution[:,0,0], T.solution[:, 1, 1], T.A_field[ti], T.E_field[ti])
        elif P.system == 'num':
            P_inter_E_dir_buf, P_inter_ortho_buf = polarization_inter_path(T.solution)
            J_intra_E_dir_buf, J_intra_ortho_buf, J_anom_ortho_buf = current_intra_path(T.solution)

        T.P_inter_E_dir[ti] += P_inter_E_dir_buf
        T.P_inter_ortho[ti] += P_inter_ortho_buf
        T.J_intra_E_dir[ti] += J_intra_E_dir_buf
        T.J_intra_ortho[ti] += J_intra_ortho_buf
        T.J_anom_ortho[ti] += J_anom_ortho_buf

def rk_integrate(t, y, kpath, S, y0, dk, \
                 dt, rhs_ode):

    k1 = rhs_ode(t,          y,          kpath, S.dipole_in_path, S.e_in_path, y0, dk)
    k2 = rhs_ode(t + 0.5*dt, y + 0.5*k1, kpath, S.dipole_in_path, S.e_in_path, y0, dk)
    k3 = rhs_ode(t + 0.5*dt, y + 0.5*k2, kpath, S.dipole_in_path, S.e_in_path, y0, dk)
    k4 = rhs_ode(t +     dt, y +     k3, kpath, S.dipole_in_path, S.e_in_path, y0, dk)

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


def write_current_emission(S, T, P):
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
    dt_out = T.t[1] - T.t[0]
    ndt_fft = (T.t.size-1)*P.factor_freq_resolution + 1
    freq = fftshift(fftfreq(ndt_fft, d=dt_out))
    gaussian_envelope = gaussian(T.t, P.alpha)

    if P.save_approx:
        I_intra_E_dir = T.J_intra_E_dir*S.kweight
        I_intra_ortho = T.J_intra_ortho*S.kweight

        I_inter_E_dir = diff(T.t, T.P_inter_E_dir)*S.kweight
        I_inter_ortho = diff(T.t, T.P_inter_ortho)*S.kweight

        I_anom_ortho  = T.J_anom_ortho*S.kweight

        # Eq. (81( SBE formalism paper
        I_deph_E_dir = 1/P.T2*T.P_inter_E_dir*S.kweight
        I_deph_ortho = 1/P.T2*T.P_inter_ortho*S.kweight

        I_E_dir = I_intra_E_dir + I_inter_E_dir
        I_ortho = I_intra_ortho + I_inter_ortho

        I_intra_plus_anom_ortho = I_intra_ortho + I_anom_ortho

        I_without_deph_E_dir = T.J_exact_E_dir - I_deph_E_dir
        I_without_deph_ortho = T.J_exact_ortho - I_deph_ortho

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

        I_approx_name = 'Iapprox_' + P.tail

        np.save(I_approx_name, [T.t, I_E_dir, I_ortho,
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
                       np.column_stack([T.t.real, I_E_dir.real, I_ortho.real,
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
        I_exact_E_dir = T.J_exact_E_dir*S.kweight
        I_exact_ortho = T.J_exact_ortho*S.kweight

        Int_exact_E_dir, Int_exact_ortho, Iw_exact_E_dir, Iw_exact_ortho = fourier_current_intensity(
                I_exact_E_dir, I_exact_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        I_exact_name = 'Iexact_' + P.tail
        np.save(I_exact_name, [T.t, I_exact_E_dir, I_exact_ortho,
                            freq/P.w, Iw_exact_E_dir, Iw_exact_ortho,
                            Int_exact_E_dir, Int_exact_ortho])

        if P.save_txt and P.factor_freq_resolution == 1:
            np.savetxt(I_exact_name + '.dat',
                    np.column_stack([T.t.real, I_exact_E_dir.real, I_exact_ortho.real,
                                        (freq/P.w).real, Iw_exact_E_dir.real, Iw_exact_E_dir.imag,
                                        Iw_exact_ortho.real, Iw_exact_ortho.imag,
                                        Int_exact_E_dir.real, Int_exact_ortho.real]),
                    header="t, I_exact_E_dir, I_exact_ortho, freqw/w, Re(Iw_exact_E_dir), Im(Iw_exact_E_dir), Re(Iw_exact_ortho), Im(Iw_exact_ortho), Int_exact_E_dir, Int_exact_ortho",
                    fmt='%+.18e')

    if P.save_latex_pdf:
        write_and_compile_latex_PDF(T.t, freq, T.E_field, T.A_field, I_exact_E_dir, I_exact_ortho, \
                Int_exact_E_dir, Int_exact_ortho, S.E_dir, S.paths, S.run_time, P)


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

