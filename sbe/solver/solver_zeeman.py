from math import ceil
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numba import njit
from scipy.integrate import ode

from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.solver import make_electric_field, make_zeeman_field, make_zeeman_field_derivative
from sbe.solver import print_user_info, BZ_plot, initial_condition, gaussian_envelope, diff, solution_containers, gaussian, write_current_emission


def sbe_zeeman_solver(sys, dipole_k, dipole_B, params):
    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_file = params.save_file
    save_full = params.save_full
    save_approx = params.save_approx
    save_txt = params.save_txt
    gauge = params.gauge

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

    B0 = params.B0*co.T_to_au                      # Magnetic field amplitude
    incident_angle = np.radians(params.incident_angle)
    mu = co.muB_to_au*np.array([params.mu_x, params.mu_y, params.mu_z])

    # Time scales
    T1 = params.T1*co.fs_to_au                     # Occupation damping time
    T2 = params.T2*co.fs_to_au                     # Polarization damping time
    gamma1 = 1/T1                                  # Occupation damping parameter
    gamma2 = 1/T2                                  # Polarization damping param

    Nf = int((abs(2*params.t0))/params.dt)
    # Find out integer times Nt fits into total time steps
    if Nf < params.Nt:
        dt_out = int(ceil(Nf/params.Nt))

    # Expand time window to fit Nf output steps
    Nt = dt_out*params.Nt
    total_fs = Nt*params.dt
    t0 = (-total_fs/2)*co.fs_to_au                 # Initial time condition
    tf = (total_fs/2)*co.fs_to_au                  # Final time
    dt = params.dt*co.fs_to_au                     #

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
        angle_inc_E_field = params.angle_in_E_field
        Nk1 = params.Nk1
        Nk2 = params.Nk2
        Nk = Nk2*Nk2

    b1 = params.b1                                 # Reciprocal lattice vectors
    b2 = params.b2

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print_user_info(BZ_type, Nk, align, angle_inc_E_field, E0, w, alpha,
                        chirp, T2, tf-t0, dt, B0=B0, mu=mu, incident_angle=incident_angle)
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
        # BZ_plot(kpnts, a, b1, b2, paths)
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                          np.sin(np.radians(angle_inc_E_field))])
        dk, kweight, _kpnts, paths = rect_mesh(params, E_dir)
        # BZ_plot(kpnts, a, b1, b2, paths)

    # Time array construction flag
    t_constructed = False

    # Initialize electric_field, create fnumba and initialize ode solver
    electric_field = make_electric_field(E0, w, alpha, chirp, phase)
    zeeman_field = make_zeeman_field(B0, mu, w, alpha, chirp, phase, E_dir,
                                     incident_angle)
    zeeman_field_derivative = \
        make_zeeman_field_derivative(B0, mu, w, alpha, chirp, phase, E_dir,
                                     incident_angle)

    fnumba = make_fnumba(sys, dipole_k, dipole_B, gamma1, gamma2, E_dir,
                         electric_field, zeeman_field, zeeman_field_derivative,
                         gauge=gauge)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method='bdf', max_step=dt)

    t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, P_E_dir, P_ortho, Zee_field =\
        solution_containers(Nk1, Nk2, params.Nt, save_approx, save_full, zeeman=True)

    # Exact emission function
    # Set after first run
    emission_exact_path = None
    # Approximate (kira & koch) emission function
    # Set after first run if save_approx=True
    current_path = None
    polarization_path = None

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    for Nk2_idx, path in enumerate(paths):
        if not save_full:
            # If we don't need the full solution only operate on path idx 0
            Nk2_idx = 0

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        m_zee = zeeman_field(t0)

        ev = sys.efjit[0](kx=kx_in_path, ky=ky_in_path, m_zee_x=m_zee[0],
                          m_zee_y=m_zee[1], m_zee_z=m_zee[2])
        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path, m_zee_x=m_zee[0],
                          m_zee_y=m_zee[1], m_zee_z=m_zee[2])
        y0 = initial_condition(e_fermi, temperature, ev, ec)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0, t0).set_f_params(path, dk, y0)

        # Propagate through time

        # Index of current integration time step
        ti = 0
        # Index of current output time step
        t_idx = 0

        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti % 1000 == 0 and user_out):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step
            if ti % dt_out == 0:
                # Do not append the last element (A_field)
                # If save_full is False Nk2_idx is 0 as only the current path
                # is saved
                solution[:, Nk2_idx, t_idx, :] = solver.y[:-1].reshape(Nk1, 4)
                # Construct time array only once
                if not t_constructed:
                    # Construct time and A_field only in first round
                    t[t_idx] = solver.t
                    A_field[t_idx] = solver.y[-1].real
                    Zee_field[t_idx] = zeeman_field(t[t_idx])

                t_idx += 1
            # Increment time counter
            ti += 1

        if not t_constructed:
            # Construct the function after the first full run!
            emission_exact_path = make_emission_exact_zeeman_path(sys, Nk1, params.Nt, E_dir,
                                                                  A_field, gauge, Zee_field)
            if save_approx:
                # Only need kira & koch formulas if save_approx is set
                current_path = make_current_zeeman_path(sys, Nk1, params.Nt, E_dir, A_field,
                                                        gauge, Zee_field)
                polarization_path = make_polarization_zeeman_path(dipole, Nk1, params.Nt, E_dir,
                                                                  A_field, gauge, Zee_field)


        #Compute per path observables
        emission_exact_path(path, solution[:, Nk2_idx, :, :], I_exact_E_dir, I_exact_ortho)

        # Flag that time array has been built up
        t_constructed = True


    # Filename tail
    tail = 'E_{:.2f}_B0_{:.3f}_mu_x{:.2f}-y{:2f}-z{:2f}_w_{:.2f}_a_{:.2f}_{}_t0_{:.2f}_NK1-{}_NK2-{}_T1_{:.2f}_T2_{:.2f}_chirp_{:.3f}_ph_{:.2f}'\
        .format(E0*co.au_to_MVpcm, B0*co.au_to_T,
                mu[0]*co.au_to_muB, mu[1]*co.au_to_muB, mu[2]*co.au_to_muB,
                w*co.au_to_THz, alpha*co.au_to_fs, gauge, params.t0, Nk1, Nk2,
                T1*co.au_to_fs, T2*co.au_to_fs, chirp*co.au_to_THz, phase)

    write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian(t, alpha), save_approx, save_txt)

    # Save the parameters of the calculation
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(params.__dict__))

    if save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution=solution, paths=paths,
                 electric_field=electric_field(t), A_field=A_field, Zee_field=Zee_field)


def emission_exact(sys, paths, tarr, solution, E_dir, A_field, zeeman_field,
                   gauge):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # n_time_steps = np.size(solution[0, 0, :, 0])
    n_time_steps = np.size(tarr)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time, t in enumerate(tarr):
        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]
        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]

            if gauge == 'length':
                kx_in_path_h_deriv = kx_in_path
                ky_in_path_h_deriv = ky_in_path

                kx_in_path_U = kx_in_path
                ky_in_path_U = ky_in_path

            elif gauge == 'velocity' or gauge == 'velocity_extra':
                kx_in_path_h_deriv = kx_in_path + A_field[i_time]*E_dir[0]
                ky_in_path_h_deriv = ky_in_path + A_field[i_time]*E_dir[1]

                kx_in_path_U = kx_in_path + A_field[i_time]*E_dir[0]
                ky_in_path_U = ky_in_path + A_field[i_time]*E_dir[1]

            h_deriv_x = evmat(sys.hderivfjit[0],
                              kx=kx_in_path_h_deriv, ky=ky_in_path_h_deriv,
                              m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
            h_deriv_y = evmat(sys.hderivfjit[1],
                              kx=kx_in_path_h_deriv, ky=ky_in_path_h_deriv,
                              m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.Uf(kx=kx_in_path_U, ky=ky_in_path_U,
                       m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
            U_h = sys.Uf_h(kx=kx_in_path_U, ky=ky_in_path_U,
                           m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

            for i_k in range(np.size(kx_in_path)):

                dH_U_E_dir = np.matmul(h_deriv_E_dir[:, :, i_k], U[:, :, i_k])
                U_h_H_U_E_dir = np.matmul(U_h[:, :, i_k], dH_U_E_dir)

                dH_U_ortho = np.matmul(h_deriv_ortho[:, :, i_k], U[:, :, i_k])
                U_h_H_U_ortho = np.matmul(U_h[:, :, i_k], dH_U_ortho)

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho


def make_fnumba(sys, dipole_k, dipole_B, gamma1, gamma2, E_dir,
                electric_field, zeeman_field, zeeman_field_derivative,
                gauge='velocity'):
    # Wire the energies
    evf = sys.efjit[0]
    ecf = sys.efjit[1]

    # Wire the dipoles
    # kx-parameter
    di_00xf = dipole_k.Axfjit[0][0]
    di_01xf = dipole_k.Axfjit[0][1]
    di_11xf = dipole_k.Axfjit[1][1]

    # ky-parameter
    di_00yf = dipole_k.Ayfjit[0][0]
    di_01yf = dipole_k.Ayfjit[0][1]
    di_11yf = dipole_k.Ayfjit[1][1]

    # Mx - Zeeman z parameter
    di_00mxf = dipole_B.Mxfjit[0][0]
    di_01mxf = dipole_B.Mxfjit[0][1]
    di_11mxf = dipole_B.Mxfjit[1][1]

    di_00myf = dipole_B.Myfjit[0][0]
    di_01myf = dipole_B.Myfjit[0][1]
    di_11myf = dipole_B.Myfjit[1][1]

    di_00mzf = dipole_B.Mzfjit[0][0]
    di_01mzf = dipole_B.Mzfjit[0][1]
    di_11mzf = dipole_B.Mzfjit[1][1]

    @njit
    def flength(t, y, kpath, dk, y0):

        # Preparing system parameters, energies, dipoles
        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]
        kx = kpath[:, 0]
        ky = kpath[:, 1]
        ev = evf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ec = ecf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ecv_in_path = ec - ev

        di_00x = di_00xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01x = di_01xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11x = di_11xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_00y = di_00yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01y = di_01yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11y = di_11yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Gradient term coefficient
        electric_f = electric_field(t)
        D = electric_f/(2*dk)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            if k == 0:
                m = 4*(k+1)
                n = 4*(Nk_path-1)
            elif k == Nk_path-1:
                m = 0
                n = 4*(k-1)
            else:
                m = 4*(k+1)
                n = 4*(k-1)

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
            x[i] = 2*(y[i+1]*wr_c).imag + D*(y[m] - y[n]) \
                - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag + D*(y[m+3] - y[n+3]) \
                - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    @njit
    def fvelocity(t, y, kpath, dk, y0):

        # Preparing system parameters, energies, dipoles

        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]
        k_shift = y[-1].real

        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ev = evf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ec = ecf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ecv_in_path = ec - ev

        di_00x = di_00xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01x = di_01xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11x = di_11xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_00y = di_00yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01y = di_01yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11y = di_11yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Gradient term coefficient
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

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    @njit
    def fvelocity_extra(t, y, kpath, dk, y0):

        # Preparing system parameters, energies, dipoles

        # k_shift caused by minimal coupling (k - A)
        k_shift = y[-1].real
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        # needed for change in energies
        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]

        ev = evf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ec = ecf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ecv_in_path = ec - ev

        # Electric
        # Electric dipoles
        di_00x = di_00xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01x = di_01xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11x = di_11xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_00y = di_00yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01y = di_01yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11y = di_11yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        electric_f = electric_field(t)
        wr_in_path = electric_f*(E_dir[0]*di_01x + E_dir[1]*di_01y)
        wr_diag_in_path = electric_f*(E_dir[0]*di_00x + E_dir[1]*di_00y \
                                   - (E_dir[0]*di_11x + E_dir[1]*di_11y))

        # Magnetic
        # Magnetic dipole integral
        di_00mx = di_00mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01mx = di_01mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11mx = di_11mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        di_00my = di_00myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01my = di_01myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11my = di_11myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        di_00mz = di_00mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01mz = di_01mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11mz = di_11mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        # Time derivative zeeman field
        mxd, myd, mzd = zeeman_field_derivative(t)
        mr_in_path = mxd*di_01mx + myd*di_01my + mzd*di_01mz
        mr_diag_in_path = mxd*(di_00mx - di_11mx) + myd*(di_00my - di_11my) \
            + mzd*(di_00mz - di_11mz)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            wr = wr_in_path[k]
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            wr_diag = wr_diag_in_path[k]

            # Magnetic frequency: M_21(k).Bdot(t)
            mr_diag = mr_diag_in_path[k]
            mr = mr_in_path[k]
            mr_c = mr.conjugate()

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(y[i+1]*(wr_c + mr_c)).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*(wr_diag + mr_diag))*y[i+1] \
                - 1j*(wr + mr)*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*(wr_c + mr_c)).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    freturn = None
    if gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
    elif gauge == 'length':
        print("Using length gauge")
        freturn = flength
    elif gauge == "velocity_extra":
        print("Using velocity extra gauge")
        freturn = fvelocity_extra
    else:
        raise AttributeError("You have to either assign length, length_extra, velocity or" +
                             "velocity_extra gauge")

    def f(t, y, kpath, dk, y0):
        return freturn(t, y, kpath, dk, y0)

    return f
