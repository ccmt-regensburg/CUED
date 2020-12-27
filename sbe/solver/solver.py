import time
import os
from math import modf
import numpy as np
from numpy.fft import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode
from scipy.sparse import lil_matrix
from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.solver import make_current_path, make_polarization_path
from sbe.solver import make_emission_exact_path_length, make_emission_exact_path_velocity
from sbe.solver import make_electric_field


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
    user_out = params.user_out                     # Command line progress output
    save_full = params.save_full                   # Save full density matrix
    save_approx = params.save_approx               # Save kira & koch approx. results
    save_txt = params.save_txt                     # Save data as human readable text file
    do_semicl = params.do_semicl                   # Additional semiclassical observable calculation
    gauge = params.gauge                           # length (dipole) or velocity (houston) gauge

    use_jacobian = False
    if hasattr(params, 'use_jacobian'):
        use_jacobian = params.use_jacobian

    symmetric_insulator = False                    # special flag for more accurate insulator calc.
    if hasattr(params, 'symmetric_insulator'):
        symmetric_insulator = params.symmetric_insulator

    method = 'bdf'
    if hasattr(params, 'solver_method'):           # 'adams' non-stiff and 'bdf' stiff problems,
        method = params.solver_method              # 'rk4' Runge-Kutta 4th order

    dk_order = 8
    if hasattr(params, 'dk_order'):                # Accuracy order of numerical density-matrix k-deriv.
        dk_order = params.dk_order                 # when using the length gauge (avail: 2,4,6,8)
        if dk_order not in [2, 4, 6, 8]: 
            quit("dk_order needs to be either 2, 4, 6, or 8.")

    # System parameters
    a = params.a                                   # Lattice spacing
    e_fermi = params.e_fermi*co.eV_to_au           # Fermi energy
    temperature = params.temperature*co.eV_to_au   # Temperature

    # Driving field parameters
    E0 = params.E0*co.MVpcm_to_au                  # Driving pulse field amplitude
    w = params.w*co.THz_to_au                      # Driving pulse frequency
    chirp = params.chirp*co.THz_to_au              # Pulse chirp frequency
    alpha = params.alpha*co.fs_to_au               # Gaussian pulse width
    phase = params.phase                           # Carrier-envelope phase

    # Time scales
    T1 = params.T1*co.fs_to_au                     # Occupation damping time
    T2 = params.T2*co.fs_to_au                     # Polarization damping time
    gamma1 = 1/T1                                  # Occupation damping parameter
    gamma2 = 1/T2                                  # Polarization damping

    Nf = int((abs(2*params.t0))/params.dt)
    if modf((2*params.t0/params.dt))[0] > 1e-12:
        print("WARNING: The time window divided by dt is not an integer.")
    # Define a proper time window if Nt exists
    # +1 assures the inclusion of tf in the calculation
    Nt = Nf + 1
    t0 = params.t0*co.fs_to_au
    tf = -t0
    dt = params.dt*co.fs_to_au

    # Brillouin zone type
    BZ_type = params.BZ_type                       # Type of Brillouin zone

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1 = params.Nk1                           # kpoints in b1 direction
        Nk2 = params.Nk2                           # kpoints in b2 direction
        Nk = Nk1*Nk2                               # Total number of kpoints
        align = params.align                       # E-field alignment
        angle_inc_E_field = None
    elif BZ_type == '2line':
        align = None
        angle_inc_E_field = params.angle_inc_E_field
        Nk1 = params.Nk1
        Nk2 = params.Nk2
        Nk = Nk1*Nk2

    b1 = params.b1                                 # Reciprocal lattice vectors
    b2 = params.b2

    # higher precision (quadruple for reducing numerical noise
    precision = 'default'
    if hasattr(params, 'precision'):
        precision = params.precision

    if precision == 'default':
        type_real_np    = np.float64
        type_complex_np = np.complex128
    elif precision == 'quadruple':
        type_real_np    = np.float128
        type_complex_np = np.complex256
        # disable numba since it doesn't support float128 and complex256
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        if method != 'rk4': quit("Error: Quadruple precision only works with Runge-Kutta 4 ODE solver.")
    else: quit("Only default or quadruple precision available.")

    print("precision =", precision)

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print_user_info(BZ_type, do_semicl, Nk, align, angle_inc_E_field, E0, w, alpha,
                        chirp, T2, tf-t0, dt, method, precision)
    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        _kpnts, paths, area = hex_mesh(Nk1, Nk2, a, b1, b2, align)
        kweight = area/Nk
        dk = 1/Nk1
        if align == 'K':
            E_dir = np.array([1, 0])
        elif align == 'M':
            E_dir = np.array([np.cos(np.radians(-30)),
                              np.sin(np.radians(-30))])
        # BZ_plot(_kpnts, a, b1, b2, paths)
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                          np.sin(np.radians(angle_inc_E_field))])
        dk, kweight, _kpnts, paths = rect_mesh(params, E_dir)
        # BZ_plot(_kpnts, a, b1, b2, paths)

    # Initialize electric_field, create fnumba and initialize ode solver
    if electric_field_function is None:
        electric_field = make_electric_field(E0, w, alpha, chirp, phase)
    else:
        electric_field = electric_field_function

    fnumba, fjac = make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field,
                               gauge=gauge, do_semicl=do_semicl, use_jacobian=use_jacobian)

    if method == 'bdf' or method == 'adams':
        solver = ode(fnumba, jac=fjac).set_integrator('zvode', method=method, max_step=dt)
    # if use_jacobian:
        # solver.set_integrator('zvode', method=method, max_step=dt, lband=2, uband=2)
    # else:
        # solver.set_integrator('zvode', method=method, max_step=dt)

    t, A_field, E_field, solution, solution_y_vec, I_exact_E_dir, I_exact_ortho, \
        J_E_dir, J_ortho, P_E_dir, P_ortho, _dummy = \
        solution_container(Nk1, Nt, save_approx, type_real_np, type_complex_np)

    # Only define full density matrix solution if save_full is True
    if save_full:
        solution_full = np.empty((Nk1, Nk2, Nt, 4), dtype=np.complex128)

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    for Nk2_idx, path in enumerate(paths):

        if gauge == 'length':
            emission_exact_path = make_emission_exact_path_length(sys, path, E_dir, do_semicl, curvature, symmetric_insulator)
        if gauge == 'velocity':
            emission_exact_path = make_emission_exact_path_velocity(sys, path, E_dir, do_semicl, curvature, symmetric_insulator)
        if save_approx:
            polarization_path = make_polarization_path(dipole, path, E_dir, gauge)
            current_path = make_current_path(sys, path, E_dir, gauge)

        print("Path: ", Nk2_idx + 1)

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        if do_semicl:
            zero_arr = np.zeros(np.size(kx_in_path), dtype=np.complex128)
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
        y0 = initial_condition(e_fermi, temperature, ev, ec)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        if method == 'bdf' or method == 'adams':
            solver.set_initial_value(y0, t0)\
                .set_f_params(path, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order)\
               .set_jac_params(path, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order)
            # if use_jacobian:
            #     solver.set_jac_params(path, dk, ecv_in_path, dipole_in_path, A_in_path)

        elif method == 'rk4':
            solution_y_vec[:] = y0

        # Propagate through time
        # Index of current integration time step
        ti = 0
        solver_successful = True

        while solver_successful and ti < Nt:
            # User output of integration progress
            if (ti % (Nt//20) == 0 and user_out):
                print('{:5.2f}%'.format((ti/Nt)*100))

            if method == 'bdf' or method == 'adams':

                # Do not append the last element (A_field)
                solution[:, :] = solver.y[:-1].reshape(Nk1, 4)

                # Construct time array only once
                if Nk2_idx == 0:
                    # Construct time and A_field only in first round
                    t[ti] = solver.t
                    A_field[ti] = solver.y[-1].real
                    E_field[ti] = electric_field(t[ti])

            elif method == 'rk4':

                # Do not append the last element (A_field)
                solution[:, :] = solution_y_vec[:-1].reshape(Nk1, 4)

                # Construct time array only once
                if Nk2_idx == 0:
                    # Construct time and A_field only in first round
                    t[ti] = ti*dt
                    A_field[ti] = solution_y_vec[-1].real
                    E_field[ti] = electric_field(t[ti])

            # Only write full density matrix solution if save_full is True
            if save_full:
                solution_full[:, Nk2_idx, ti, :] = solution

            I_E_dir_buf, I_ortho_buf = emission_exact_path(solution, E_field[ti], A_field[ti])
            I_exact_E_dir[ti] += I_E_dir_buf
            I_exact_ortho[ti] += I_ortho_buf
            if save_approx:
                P_E_dir_buf, P_ortho_buf = polarization_path(solution[:, 2], A_field[ti])
                P_E_dir[ti] += P_E_dir_buf
                P_ortho[ti] += P_ortho_buf
                J_E_dir_buf, J_ortho_buf = current_path(solution[:, 0], solution[:, 3], A_field[ti])
                J_E_dir[ti] += J_E_dir_buf
                J_ortho[ti] += J_ortho_buf

            if method == 'bdf' or method == 'adams':
                # Integrate one integration time step
                solver.integrate(solver.t + dt)
                solver_successful = solver.successful()

            elif method == 'rk4':
                solution_y_vec = rk_integrate(t[ti], solution_y_vec, path, dk, ecv_in_path, \
                                              dipole_in_path, A_in_path, y0, dk_order)

            # Increment time counter
            ti += 1

    # End time of solver loop
    end_time = time.perf_counter()
    # breakpoint()

    # Write solutions
    # Filename tail
    tail = 'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}_solver_{:s}'\
        .format(E0*co.au_to_MVpcm, w*co.au_to_THz, alpha*co.au_to_fs, gauge, params.t0, params.dt, Nk1, Nk2, T1*co.au_to_fs, T2*co.au_to_fs, chirp*co.au_to_THz, phase, method)

    write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian(t, alpha), save_approx, save_txt)

    # Save the parameters of the calculation
    run_time = end_time - start_time
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(params.__dict__) + "\n\n")
    paramsfile.write("Runtime: {:.16f} s".format(run_time))
    paramsfile.close()

    if save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution_full=solution_full, paths=paths,
                 electric_field=electric_field(t), A_field=A_field)


def make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field, gauge,
                do_semicl, use_jacobian):
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

    @njit
    def flength(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

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
            elif k == 1:
                left2  = 4*(Nk_path-1)
                left3  = 4*(Nk_path-2)            
                left4  = 4*(Nk_path-3)          
            elif k == 2:
                left3  = 4*(Nk_path-1)          
                left4  = 4*(Nk_path-2) 
            elif k == 3:
                left4  = 4*(Nk_path-1) 
            elif k == Nk_path-1:
                right4 = 4*3
                right3 = 4*2
                right2 = 4*1
                right  = 4*0
            elif k == Nk_path-2:
                right4 = 4*2
                right3 = 4*1
                right2 = 4*0
            elif k == Nk_path-3:
                right4 = 4*1
                right3 = 4*0
            elif k == Nk_path-4:
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

    @njit
    def pre_velocity(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if do_semicl:
            zero_arr = np.zeros(kx.size, dtype=np.complex128)
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

    @njit
    def fvelocity(t, y, kpath, _dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        ecv_in_path, dipole_in_path, A_in_path = pre_velocity(kpath, y[-1].real)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

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

    @njit
    def jac_velocity(t, y, kpath, _dk, ecv_in_path, dipole_in_path, A_in_path, _y0):
        """
        Jacobian of SBE in the velocity gauge
        """
        ecv_in_path, dipole_in_path, A_in_path = pre_velocity(kpath, y[-1])

        # Empty Jacobian, packed format
        # Read https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
        # definition of uband in 'vode'
        # J[i-j+uband, j] = J[i, j]; uband=2
        dim = np.shape(y)[0]
        J = np.zeros((dim, dim), dtype=np.complex128)

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

            J[i, i]   = -gamma1
            # J[2, i] = -gamma1
            J[i, i+1] = -1j*wr_c
            # J[1, i+1] = -1j*wr_c
            J[i, i+2] = 1j*wr
            # J[0, i+2] = 1j*wr
            # J[i, i+3] is zero

            J[i+1, i]   = -1j*wr_c
            # J[3, i]   = -1j*wr_c
            J[i+1, i+1] = -1j*ecv - gamma2 + 1j*wr_d_diag
            # J[2, i+1] = -1j*ecv - gamma2 + 1j*wr_d_diag
            # J[i+1, i+2] is zero
            J[i+1, i+3] = 1j*wr
            # J[0, i+3] = 1j*wr

            J[i+2, i]   = 1j*wr_c
            # J[4, i]   = 1j*wr_c
            # J[i+2, i+1] is zero
            J[i+2, i+2] = 1j*ecv - gamma2 - 1j*wr_d_diag
            # J[2, i+2] = 1j*ecv - gamma2 - 1j*wr_d_diag
            J[i+2, i+3] = -1j*wr_c
            # J[1, i+3] = -1j*wr_c

            # J[i+3, i] is zero
            J[i+3, i+1] = 1j*wr_c
            # J[4, i+1] = 1j*wr_c
            J[i+3, i+2] = -1j*wr
            # J[3, i+2] = -1j*wr
            J[i+3, i+3] = -gamma1
            # J[2, i+3] = -gamma1

        return J

    freturn = None
    fjac_return = None
    if gauge == 'length':
        print("Using length gauge")
        freturn = flength
        if use_jacobian:
            fjac_return = None
    elif gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
        if use_jacobian:
            fjac_return = jac_velocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")


    # The python solver does not directly accept jitted functions so we wrap it
    def f(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order):
        return freturn(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order)

    if fjac_return is not None:
        def fjac(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
            return fjac_return(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)
    else:
        fjac = None

    return f, fjac

def rk_integrate(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0, dk_order):



    return y

def solution_container(Nk1, Nt, save_approx, type_real_np, type_complex_np, zeeman=False):
    """
        Function that builds the containers on which the solutions of the SBE,
        as well as the currents will be written
    """
    # Solution containers
    t = np.zeros(Nt)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
#    solution = np.zeros((Nk1, 4), dtype=np.complex128)
    solution = np.zeros((Nk1, 4), dtype=type_complex_np)

    # For hand-made Runge-Kutta method, we need the solution as array with 
    # a single index
    solution_y_vec = np.zeros((4*Nk1+1), dtype=type_complex_np)

    A_field = np.zeros(Nt, dtype=np.float64)
    E_field = np.zeros(Nt, dtype=np.float64)

    I_exact_E_dir = np.zeros(Nt, dtype=np.float64)
    I_exact_ortho = np.zeros(Nt, dtype=np.float64)

    if save_approx:
        J_E_dir = np.zeros(Nt, dtype=np.float64)
        J_ortho = np.zeros(Nt, dtype=np.float64)
        P_E_dir = np.zeros(Nt, dtype=np.float64)
        P_ortho = np.zeros(Nt, dtype=np.float64)
    else:
        J_E_dir = None
        J_ortho = None
        P_E_dir = None
        P_ortho = None

    if zeeman:
        Zee_field = np.zeros((Nt, 3), dtype=np.float64)
        return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
            P_E_dir, P_ortho, Zee_field

    return t, A_field, E_field, solution, solution_y_vec, I_exact_E_dir, I_exact_ortho, \
        J_E_dir, J_ortho, P_E_dir, P_ortho, None


def initial_condition(e_fermi, temperature, ev, ec):
    '''
    Occupy conduction band according to inital Fermi energy and temperature
    '''
    knum = ec.size
    zero_arr = np.zeros(knum, dtype=np.complex128)
    distrib_ec = np.zeros(knum, dtype=np.complex128)
    distrib_ev = np.zeros(knum, dtype=np.complex128)
    if temperature > 1e-5:
        distrib_ec += 1/(np.exp((ec-e_fermi)/temperature) + 1)
        distrib_ev += 1/(np.exp((ev-e_fermi)/temperature) + 1)
        return np.array([distrib_ev, zero_arr, zero_arr, distrib_ec]).flatten('F')

    smaller_e_fermi_ev = (e_fermi - ev) > 0
    smaller_e_fermi_ec = (e_fermi - ec) > 0

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


def write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian_envelope, save_approx, save_txt):
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
    if save_approx:
        # Only do approximate emission fourier transforms if save_approx is set
        I_E_dir = kweight*(diff(t, P_E_dir) + J_E_dir)
        I_ortho = kweight*(diff(t, P_ortho) + J_ortho)

        I_intra_E_dir = J_E_dir*kweight
        I_intra_ortho = J_ortho*kweight

        I_inter_E_dir = diff(t, P_E_dir)*kweight
        I_inter_ortho = diff(t, P_ortho)*kweight

        Iw_E_dir = fourier(dt_out, I_E_dir*gaussian_envelope)
        Iw_ortho = fourier(dt_out, I_ortho*gaussian_envelope)

        Iw_intra_E_dir = fourier(dt_out, I_intra_E_dir*gaussian_envelope)
        Iw_intra_ortho = fourier(dt_out, I_intra_ortho*gaussian_envelope)

        Iw_inter_E_dir = fourier(dt_out, I_inter_E_dir*gaussian_envelope)
        Iw_inter_ortho = fourier(dt_out, I_inter_ortho*gaussian_envelope)

        # Approximate Emission intensity
        Int_E_dir = prefac_emission*(freq**2)*np.abs(Iw_E_dir)**2
        Int_ortho = prefac_emission*(freq**2)*np.abs(Iw_ortho)**2

        Int_intra_E_dir = prefac_emission*(freq**2)*np.abs(Iw_intra_E_dir)**2
        Int_intra_ortho = prefac_emission*(freq**2)*np.abs(Iw_intra_ortho)**2

        Int_inter_E_dir = prefac_emission*(freq**2)*np.abs(Iw_inter_E_dir)**2
        Int_inter_ortho = prefac_emission*(freq**2)*np.abs(Iw_inter_ortho)**2

        I_approx_name = 'Iapprox_' + tail

        np.save(I_approx_name, [t, I_E_dir, I_ortho,
                                freq/w, Iw_E_dir, Iw_ortho,
                                Int_E_dir, Int_ortho,
                                I_intra_E_dir, I_intra_ortho,
                                Int_intra_E_dir, Int_intra_ortho,
                                I_inter_E_dir, I_inter_ortho,
                                Int_inter_E_dir, Int_inter_ortho])

        if save_txt:
            np.savetxt(I_approx_name + '.dat',
                       np.column_stack([t.real, I_E_dir.real, I_ortho.real,
                                        (freq/w).real, Iw_E_dir.real, Iw_E_dir.imag, Iw_ortho.real, Iw_ortho.imag,
                                        Int_E_dir.real, Int_ortho.real]),
                       header="t, I_E_dir, I_ortho, freqw/w, Re(Iw_E_dir), Im(Iw_E_dir), Re(Iw_ortho), Im(Iw_ortho), Int_E_dir, Int_ortho",
                       fmt='%+.18e')

    ##############################################################
    # Always calculate exact emission formula
    ##############################################################
    # kweight is different for 2line and full
    I_exact_E_dir *= kweight
    I_exact_ortho *= kweight

    Iw_exact_E_dir = fourier(dt_out, I_exact_E_dir*gaussian_envelope)
    Iw_exact_ortho = fourier(dt_out, I_exact_ortho*gaussian_envelope)
    Int_exact_E_dir = prefac_emission*(freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = prefac_emission*(freq**2)*np.abs(Iw_exact_ortho)**2

    I_exact_name = 'Iexact_' + tail
    np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])
    if save_txt:
        np.savetxt(I_exact_name + '.dat',
                   np.column_stack([t.real, I_exact_E_dir.real, I_exact_ortho.real,
                                    (freq/w).real, Iw_exact_E_dir.real, Iw_exact_E_dir.imag, Iw_exact_ortho.real, Iw_exact_ortho.imag,
                                    Int_exact_E_dir.real, Int_exact_ortho.real]),
                   header="t, I_exact_E_dir, I_exact_ortho, freqw/w, Re(Iw_exact_E_dir), Im(Iw_exact_E_dir), Re(Iw_exact_ortho), Im(Iw_exact_ortho), Int_exact_E_dir, Int_exact_ortho",
                   fmt='%+.18e')


def print_user_info(BZ_type, do_semicl, Nk, align, angle_inc_E_field, E0, w, alpha, chirp,
                    T2, tfmt0, dt, method, precision, B0=None, mu=None, incident_angle=None):
    """
        Function that prints the input parameters if usr_info = True
    """
    print("Input parameters:")
    print("Brillouin zone                  = " + BZ_type)
    print("Do Semiclassics                 = " + str(do_semicl))
    print("ODE solver method               = " + str(method))
    print("Precision (default = double)    = " + str(precision))
    print("Number of k-points              = " + str(Nk))
    if BZ_type == 'full':
        print("Driving field alignment         = " + align)
    elif BZ_type == '2line':
        print("Driving field direction         = " + str(angle_inc_E_field))
    if B0 is not None:
        print("Incident angle                  = " + str(np.rad2deg(incident_angle)))
    print("Driving amplitude (MV/cm)[a.u.] = " + "("
          + '{:.6f}'.format(E0*co.au_to_MVpcm) + ")"
          + "[" + '{:.6f}'.format(E0) + "]")
    if B0 is not None:
        print("Magnetic amplitude (T)[a.u.]    = " + "("
              + '%.6f'%(B0*co.au_to_T) + ")"
              + "[" + '%.6f'%(B0) + "]")
        print("Magnetic moments ", mu)
    print("Pulse Frequency (THz)[a.u.]     = " + "("
          + '{:.6f}'.format(w*co.au_to_THz) + ")"
          + "[" + '{:.6f}'.format(w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "("
          + '{:.6f}'.format(alpha*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(alpha) + "]")
    print("Chirp rate (THz)[a.u.]          = " + "("
          + '{:.6f}'.format(chirp*co.au_to_THz) + ")"
          + "[" + '{:.6f}'.format(chirp) + "]")
    print("Damping time (fs)[a.u.]         = " + "("
          + '{:.6f}'.format(T2*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(T2) + "]")
    print("Total time (fs)[a.u.]           = " + "("
          + '{:.6f}'.format(tfmt0*co.au_to_fs) + ")"
          + "[" + '{:.5f}'.format(tfmt0) + "]")
    print("Time step (fs)[a.u.]            = " + "("
          + '{:.6f}'.format(dt*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(dt) + "]")


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
