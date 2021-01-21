import time
import os
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.utility import conditional_njit
from sbe.solver import make_current_path, make_polarization_path
from sbe.solver import make_emission_exact_path_length, make_emission_exact_path_velocity
from sbe.solver import make_electric_field, parse_params


def sbe_solver(sys, dipole, params, curvature, electric_field_function=None):
    """
        Solver for the semiconductor bloch equation ( eq. (39) or (47) in https://arxiv.org/abs/2008.03177)
        for a two band system with analytical calculation of the dipole elements

        Author:
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
        electric_field_function : function
            Jitted function of a user provided electric field.
            Can only take time as parameter. If is None takes
            electric field from sbe/solver/fields.py

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

    # Initialize electric_field, create fnumba and initialize ode solver
    if electric_field_function is None:
        electric_field = make_electric_field(P.E0, P.w, P.alpha, P.chirp, P.phase, P.type_real_np)
    else:
        electric_field = electric_field_function

    fnumba = make_fnumba(sys, dipole, E_dir, P.gamma1, P.gamma2, P.dk_order, electric_field,
                         P.gauge, P.type_complex_np, P.do_semicl)

    if P.solver_method in ('bdf', 'adams'):
        solver = ode(fnumba).set_integrator('zvode', method=P.solver_method, max_step=P.dt)

    t, A_field, E_field, solution, solution_y_vec, I_exact_E_dir, I_exact_ortho, \
        J_E_dir, J_ortho, P_E_dir, P_ortho, J_anom_ortho, _dummy = \
        solution_container(P)

    # Only define full density matrix solution if save_full is True
    if P.save_full:
        solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, 4), dtype=P.type_complex_np)

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    for Nk2_idx, path in enumerate(paths):

        # parallelization if requested in runscript
        if P.Nk2_idx_ext != Nk2_idx and P.Nk2_idx_ext >= 0: continue

        if P.gauge == 'length':
            emission_exact_path = make_emission_exact_path_length(sys, path, E_dir, curvature, P)
        if P.gauge == 'velocity':
            emission_exact_path = make_emission_exact_path_velocity(sys, path, E_dir, curvature, P)
        if P.save_approx:
            polarization_path = make_polarization_path(dipole, path, E_dir, P)
            current_path = make_current_path(sys, path, E_dir, curvature, P)

        print("Path: ", Nk2_idx + 1)

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        if P.do_semicl:
            zero_arr = np.zeros(np.size(kx_in_path), dtype=P.type_complex_np)
            dipole_in_path = zero_arr
            A_in_path = zero_arr
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
            # A[0, 1, :] means 0-1 offdiagonal element
            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        ev = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)
        ecv_in_path = ec - ev

        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = initial_condition(ev, ec, P)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        if P.solver_method in ('bdf', 'adams'):
            solver.set_initial_value(y0, P.t0)\
                .set_f_params(path, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

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
                solution[:, :] = solver.y[:-1].reshape(P.Nk1, 4)

                # Construct time array only once
                if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
                    # Construct time and A_field only in first round
                    t[ti] = solver.t
                    A_field[ti] = solver.y[-1].real
                    E_field[ti] = electric_field(t[ti])

            elif P.solver_method == 'rk4':
                # Do not append the last element (A_field)
                solution[:, :] = solution_y_vec[:-1].reshape(P.Nk1, 4)

                # Construct time array only once
                if Nk2_idx == 0 or P.Nk2_idx_ext > 0:
                    # Construct time and A_field only in first round
                    t[ti] = ti*P.dt + P.t0
                    A_field[ti] = solution_y_vec[-1].real
                    E_field[ti] = electric_field(t[ti])

            # Only write full density matrix solution if save_full is True
            if P.save_full:
                solution_full[:, Nk2_idx, ti, :] = solution

            I_E_dir_buf, I_ortho_buf = emission_exact_path(solution, E_field[ti], A_field[ti])
            I_exact_E_dir[ti] += I_E_dir_buf
            I_exact_ortho[ti] += I_ortho_buf
            if P.save_approx:
                P_E_dir_buf, P_ortho_buf = polarization_path(solution[:, 2], A_field[ti])
                P_E_dir[ti] += P_E_dir_buf
                P_ortho[ti] += P_ortho_buf
                J_E_dir_buf, J_ortho_buf, J_anom_ortho_buf = current_path(solution[:, 0], solution[:, 3], A_field[ti], E_field[ti])
                J_E_dir[ti] += J_E_dir_buf
                J_ortho[ti] += J_ortho_buf
                J_anom_ortho[ti] += J_anom_ortho_buf

            if P.solver_method in ('bdf', 'adams'):
                # Integrate one integration time step
                solver.integrate(solver.t + P.dt)
                solver_successful = solver.successful()
            elif P.solver_method == 'rk4':
                solution_y_vec = rk_integrate(t[ti], solution_y_vec, path, dk, ecv_in_path, \
                                              dipole_in_path, A_in_path, y0, P.dt, fnumba)

            # Increment time counter
            ti += 1

    # End time of solver loop
    end_time = time.perf_counter()

    # Write solutions
    # Filename tail
    tail = 'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}_solver_{:s}'\
        .format(P.E0_MVpcm, P.w_THz, P.alpha_fs, P.gauge, P.t0_fs, P.dt_fs, P.Nk1, P.Nk2, P.T1_fs, P.T2_fs, P.chirp_THz, P.phase, P.solver_method)

    write_current_emission(tail, kweight, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho, J_anom_ortho, P)

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


def make_fnumba(sys, dipole, E_dir, gamma1, gamma2, dk_order, electric_field, gauge, type_complex_np,
                do_semicl):
    """
        Initialization of the solver for the sbe ( eq. (39/47/80) in https://arxiv.org/abs/2008.03177)

        Author:
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters
        ----------
        sys : class
            Symbolic Hamiltonian of the system
        dipole : class
            Symbolic expression for the dipole elements (eq. (37/38))
        E_dir : np.ndarray
            2-dimensional array with the x and y component of the electric field
        gamma1 : float
            inverse of occupation damping time (T_1 in (eq. (?))
        gamma2 : float
            inverse of polarization damping time (T_2 in eq. (80))
        electric_field : jitted function
            absolute value of the instantaneous driving field E(t) (eq. (75))
        gauge: 'length' or 'velocity'
            parameter to determine which gauge is used in the routine
        do_semicl: boolean
            parameter to determine whether a semiclassical calculation will be done

        Returns
        -------
        f :
            right hand side of ode d/dt(rho(t)) = f(rho, t) (eq. (39/47/80))
    """
    ########################################
    # Wire the energies
    ########################################
    evf = sys.efjit[0]
    ecf = sys.efjit[1]

    ########################################
    # Wire the dipoles
    ########################################
    # kx-parameter
    di_00xf = dipole.Axfjit[0][0]
    di_01xf = dipole.Axfjit[0][1]
    di_11xf = dipole.Axfjit[1][1]

    # ky-parameter
    di_00yf = dipole.Ayfjit[0][0]
    di_01yf = dipole.Ayfjit[0][1]
    di_11yf = dipole.Ayfjit[1][1]

    @conditional_njit(type_complex_np)
    def flength(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        # Gradient term coefficient
        electric_f = electric_field(t)
        D = electric_f/dk

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            right4 = 4*(k+4)
            right3 = 4*(k+3)
            right2 = 4*(k+2)
            right  = 4*(k+1)
            left   = 4*(k-1)
            left2  = 4*(k-2)
            left3  = 4*(k-3)
            left4  = 4*(k-4)
            if k == 0:
                left   = 4*(Nk_path-1)
                left2  = 4*(Nk_path-2)
                left3  = 4*(Nk_path-3)
                left4  = 4*(Nk_path-4)
            elif k == 1 and dk_order >= 4:
                left2  = 4*(Nk_path-1)
                left3  = 4*(Nk_path-2)
                left4  = 4*(Nk_path-3)
            elif k == 2 and dk_order >= 6:
                left3  = 4*(Nk_path-1)
                left4  = 4*(Nk_path-2)
            elif k == 3 and dk_order >= 8:
                left4  = 4*(Nk_path-1)
            elif k == Nk_path-1:
                right4 = 4*3
                right3 = 4*2
                right2 = 4*1
                right  = 4*0
            elif k == Nk_path-2 and dk_order >= 4:
                right4 = 4*2
                right3 = 4*1
                right2 = 4*0
            elif k == Nk_path-3 and dk_order >= 6:
                right4 = 4*1
                right3 = 4*0
            elif k == Nk_path-4 and dk_order >= 8:
                right4 = 4*0

            # Energy gap e_2(k) - e_1(k) >= 0 at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = q*d_12(k)*E(t)
            # Rabi frequency conjugate: w_R_c = q*d_21(k)*E(t)
            wr = dipole_in_path[k]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = q*(d_11(k) - d_22(k))*E(t)
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i]   = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

            # compute drift term via k-derivative
            if dk_order == 2:
                x[i]   += D*( y[right]/2   - y[left]/2  )
                x[i+1] += D*( y[right+1]/2 - y[left+1]/2 )
                x[i+3] += D*( y[right+3]/2 - y[left+3]/2 )
            elif dk_order == 4:
                x[i]   += D*(- y[right2]/12   + 2/3*y[right]   - 2/3*y[left]   + y[left2]/12 )
                x[i+1] += D*(- y[right2+1]/12 + 2/3*y[right+1] - 2/3*y[left+1] + y[left2+1]/12 )
                x[i+3] += D*(- y[right2+3]/12 + 2/3*y[right+3] - 2/3*y[left+3] + y[left2+3]/12 )
            elif dk_order == 6:
                x[i]   += D*(  y[right3]/60   - 3/20*y[right2]   + 3/4*y[right] \
                             - y[left3]/60    + 3/20*y[left2]    - 3/4*y[left] )
                x[i+1] += D*(  y[right3+1]/60 - 3/20*y[right2+1] + 3/4*y[right+1] \
                             - y[left3+1]/60  + 3/20*y[left2+1]  - 3/4*y[left+1] )
                x[i+3] += D*(  y[right3+3]/60 - 3/20*y[right2+3] + 3/4*y[right+3] \
                             - y[left3+3]/60  + 3/20*y[left2+3]  - 3/4*y[left+3] )
            elif dk_order == 8:
                x[i]   += D*(- y[right4]/280   + 4/105*y[right3]   - 1/5*y[right2]   + 4/5*y[right] \
                             + y[left4] /280   - 4/105*y[left3]    + 1/5*y[left2]    - 4/5*y[left] )
                x[i+1] += D*(- y[right4+1]/280 + 4/105*y[right3+1] - 1/5*y[right2+1] + 4/5*y[right+1] \
                             + y[left4+1] /280 - 4/105*y[left3+1]  + 1/5*y[left2+1]  - 4/5*y[left+1] )
                x[i+3] += D*(- y[right4+3]/280 + 4/105*y[right3+3] - 1/5*y[right2+3] + 4/5*y[right+3] \
                             + y[left4+3] /280 - 4/105*y[left3+3]  + 1/5*y[left2+3]  - 4/5*y[left+3] )

            x[i+2] = x[i+1].conjugate()

        x[-1] = -electric_f
        return x

    @conditional_njit(type_complex_np)
    def pre_velocity(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if do_semicl:
            zero_arr = np.zeros(kx.size, dtype=type_complex_np)
            dipole_in_path = zero_arr
            A_in_path = zero_arr
        else:
            di_00x = di_00xf(kx=kx, ky=ky)
            di_01x = di_01xf(kx=kx, ky=ky)
            di_11x = di_11xf(kx=kx, ky=ky)
            di_00y = di_00yf(kx=kx, ky=ky)
            di_01y = di_01yf(kx=kx, ky=ky)
            di_11y = di_11yf(kx=kx, ky=ky)

            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        return ecv_in_path, dipole_in_path, A_in_path

    @conditional_njit(type_complex_np)
    def fvelocity(t, y, kpath, _dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        ecv_in_path, dipole_in_path, A_in_path = pre_velocity(kpath, y[-1].real)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        electric_f = electric_field(t)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f

        return x

    freturn = None
    if gauge == 'length':
        print("Using length gauge")
        freturn = flength
    elif gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")


    # The python solver does not directly accept jitted functions so we wrap it
    def f(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        return freturn(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

    return f

def rk_integrate(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0, \
                 dt, fnumba):

    k1 = fnumba(t,          y,          kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)
    k2 = fnumba(t + 0.5*dt, y + 0.5*k1, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)
    k3 = fnumba(t + 0.5*dt, y + 0.5*k2, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)
    k4 = fnumba(t +     dt, y +     k3, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

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
    solution = np.zeros((P.Nk1, 4), dtype=P.type_complex_np)

    # For hand-made Runge-Kutta method, we need the solution as array with
    # a single index
    solution_y_vec = np.zeros((4*P.Nk1+1), dtype=P.type_complex_np)

    A_field = np.zeros(P.Nt, dtype=P.type_real_np)
    E_field = np.zeros(P.Nt, dtype=P.type_real_np)

    I_exact_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
    I_exact_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

    if P.save_approx:
        J_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        J_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
        P_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        P_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
        J_anom_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
    else:
        J_E_dir = None
        J_ortho = None
        P_E_dir = None
        P_ortho = None
        J_anom_ortho = None

    if zeeman:
        Zee_field = np.zeros((P.Nt, 3), dtype=P.type_real_np)
        return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
            P_E_dir, P_ortho, Zee_field

    return t, A_field, E_field, solution, solution_y_vec, I_exact_E_dir, I_exact_ortho, \
        J_E_dir, J_ortho, P_E_dir, P_ortho, J_anom_ortho, None


def initial_condition(ev, ec, P):
    '''
    Occupy conduction band according to inital Fermi energy and temperature
    '''
    knum = ec.size
    zero_arr = np.zeros(knum, dtype=P.type_complex_np)
    distrib_ec = np.zeros(knum, dtype=P.type_complex_np)
    distrib_ev = np.zeros(knum, dtype=P.type_complex_np)
    if P.temperature > 1e-5:
        distrib_ec += 1/(np.exp((ec-P.e_fermi)/P.temperature) + 1)
        distrib_ev += 1/(np.exp((ev-P.e_fermi)/P.temperature) + 1)
        return np.array([distrib_ev, zero_arr, zero_arr, distrib_ec]).flatten('F')

    smaller_e_fermi_ev = (P.e_fermi - ev) > 0
    smaller_e_fermi_ec = (P.e_fermi - ec) > 0

    distrib_ev[smaller_e_fermi_ev] += 1
    distrib_ec[smaller_e_fermi_ec] += 1
    return np.array([distrib_ev, zero_arr, zero_arr, distrib_ec]).flatten('F')


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
    freq = fftshift(fftfreq(t.size, d=dt_out))
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
    # kweight is different for rectangle and full
    if P.save_exact:
        I_exact_E_dir *= kweight
        I_exact_ortho *= kweight

        Int_exact_E_dir, Int_exact_ortho, Iw_exact_E_dir, Iw_exact_ortho = fourier_current_intensity(
                I_exact_E_dir, I_exact_ortho, gaussian_envelope, dt_out, prefac_emission, freq)

        I_exact_name = 'Iexact_' + tail
        np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho,
                            freq/P.w, Iw_exact_E_dir, Iw_exact_ortho,
                            Int_exact_E_dir, Int_exact_ortho])
        if P.save_txt:
            np.savetxt(I_exact_name + '.dat',
                    np.column_stack([t.real, I_exact_E_dir.real, I_exact_ortho.real,
                                        (freq/P.w).real, Iw_exact_E_dir.real, Iw_exact_E_dir.imag,
                                        Iw_exact_ortho.real, Iw_exact_ortho.imag,
                                        Int_exact_E_dir.real, Int_exact_ortho.real]),
                    header="t, I_exact_E_dir, I_exact_ortho, freqw/w, Re(Iw_exact_E_dir), Im(Iw_exact_E_dir), Re(Iw_exact_ortho), Im(Iw_exact_ortho), Int_exact_E_dir, Int_exact_ortho",
                    fmt='%+.18e')


def fourier_current_intensity(I_E_dir, I_ortho, gaussian_envelope, dt_out, prefac_emission, freq):

    Iw_E_dir = fourier(dt_out, I_E_dir*gaussian_envelope)
    Iw_ortho = fourier(dt_out, I_ortho*gaussian_envelope)
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


def BZ_plot(kpnts, paths, P, si_units=False):
    """
        Function that plots the brillouin zone
    """
    if si_units:
        a = P.a_angs
        kpnts *= co.as_to_au
        b1 = P.b1_dangs
        b2 = P.b2_dangs
    else:
        a = P.a
        b1 = P.b1
        b2 = P.b2

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
    plt.scatter(kpnts[:, 0], kpnts[:, 1], s=0.1)
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
            plt.plot(co.as_to_au*path[:, 0], co.as_to_au*path[:, 1], lw=0.1)
        else:
            plt.plot(path[:, 0], path[:, 1], lw=0.1)

    plt.show()
