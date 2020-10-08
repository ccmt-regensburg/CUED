from math import ceil
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.solver import make_electric_field, make_zeeman_field, make_zeeman_field_derivative


def sbe_zeeman_solver(sys, dipole_k, dipole_B, params):
    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_file = params.save_file
    save_full = params.save_full
    save_approx = params.save_approx
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
    dt_out = int(ceil(Nf/params.Nt))

    # Expand time window to fit Nf output steps
    Nt = dt_out*params.Nt
    total_fs = Nt*params.dt
    t0 = (-total_fs/2)*co.fs_to_au                 # Initial time condition
    tf = (total_fs/2)*co.fs_to_au                  # Final time
    dt = params.dt*co.fs_to_au                     #
    dt_out = 1/(2*params.dt)                       # Solution output time step

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
        Nk_in_path = params.Nk_in_path
        Nk = 2*Nk_in_path
        align = None
        angle_inc_E_field = params.angle_inc_E_field
        Nk1 = Nk_in_path
        Nk2 = 2

    b1 = params.b1                                 # Reciprocal lattice vectors
    b2 = params.b2

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print_user_info(BZ_type, Nk, align, angle_inc_E_field, E0, B0, w, alpha,
                        chirp, T2, tf-t0, dt)
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
        dk, _kpnts, paths = rect_mesh(params, E_dir)
        # BZ_plot(kpnts, a, b1, b2, paths)

    # Time array construction flag
    t_constructed = False

    # Solution containers
    t = np.empty(params.Nt)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    if save_full:
        # Make container for full solution if it is needed
        solution = np.empty((Nk1, Nk2, params.Nt, 4), dtype=complex)
    else:
        # Only one path needed at a time if no full solution is needed
        solution = np.empty((Nk1, 1, params.Nt, 4), dtype=complex)

    A_field = np.empty(params.Nt, dtype=np.float64)
    Zee_field = np.empty((params.Nt, 3), dtype=np.float64)

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

    # Exact emission function will be set after end of first run
    emission_exact_path = None
    I_exact_E_dir = np.zeros(params.Nt, dtype=np.float64)
    I_exact_ortho = np.zeros(params.Nt, dtype=np.float64)

    # Approximate (kira & koch) containers
    if save_approx:
        current_path = None
        J_E_dir = np.zeros(params.Nt, dtype=np.float64)
        J_ortho = np.zeros(params.Nt, dtype=np.float64)
        polarization_path = None
        P_E_dir = np.zeros(params.Nt, dtype=np.float64)
        P_ortho = np.zeros(params.Nt, dtype=np.float64)
    else:
        current_path = None
        J_E_dir = None
        J_ortho = None
        polarization_path = None
        P_E_dir = None
        P_ortho = None

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

        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path, m_zee_x=m_zee[0],
                          m_zee_y=m_zee[1], m_zee_z=m_zee[2])
        y0 = initial_condition(e_fermi, temperature, ec)
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
            emission_exact_zeeman_path = make_emission_exact_zeeman_path(sys, Nk1, params.Nt, E_dir,
                                                                          A_field, gauge, Zee_field)
            if save_approx:
                # Only need kira & koch formulas if save_approx is set
                current_zeeman_path = make_current_zeeman_path(sys, Nk1, params.Nt, E_dir, A_field,
                                                               gauge, Zee_field)
                polarization_zeeman_path = make_polarization_zeeman_path(dipole, Nk1, params.Nt, E_dir,
                                                                         A_field, gauge, Zee_field)


        #Compute per path observables
        emission_exact_zeeman_path(path, solution[:, Nk2_idx, :, :], I_exact_E_dir, I_exact_ortho)

        # Flag that time array has been built up
        t_constructed = True


    # Filename tail
    tail = 'E_{:.2f}_w_{:.2f}_a_{:.2f}_{}_t0_{:.2f}_NK1-{}_NK2-{}_T1_{:.2f}_T2_{:.2f}_chirp_{:.3f}_ph_{:.2f}'\
        .format(E0*co.au_to_MVpcm, w*co.au_to_THz, alpha*co.au_to_fs, gauge, params.t0, Nk1, Nk2, T1*co.au_to_fs, T2*co.au_to_fs, chirp*co.au_to_THz, phase)

    # Fourier transforms
    dt_out = t[1] - t[0]
    freq = fftshift(fftfreq(np.size(t), d=dt_out))
    if save_approx:
        # Only do kira & koch emission fourier transforms if save_approx is set
        # Approximate emission in time
        I_E_dir = diff(t, P_E_dir)*gaussian_envelope(t, alpha) \
            + J_E_dir*gaussian_envelope(t, alpha)
        I_ortho = diff(t, P_ortho)*gaussian_envelope(t, alpha) \
            + J_ortho*gaussian_envelope(t, alpha)
        if BZ_type == '2line':
            I_E_dir *= (dk/(4*np.pi))
            I_ortho *= (dk/(4*np.pi))
        if BZ_type == 'full':
            I_E_dir *= kweight
            I_ortho *= kweight


        Iw_E_dir = fftshift(fft(I_E_dir, norm='ortho'))
        Iw_ortho = fftshift(fft(I_ortho, norm='ortho'))

        # Approximate Emission intensity
        Int_E_dir = (freq**2)*np.abs(Iw_E_dir)**2
        Int_ortho = (freq**2)*np.abs(Iw_ortho)**2

        I_approx_name = 'Iapprox_' + tail
        np.save(I_approx_name, [t, I_E_dir, I_ortho,
                                freq/w, Iw_E_dir, Iw_ortho,
                                Int_E_dir, Int_ortho])

    ##############################################################
    # Always calculate exact emission formula
    ##############################################################
    if BZ_type == '2line':
        I_exact_E_dir *= (dk/(4*np.pi))
        I_exact_ortho *= (dk/(4*np.pi))
    if BZ_type == 'full':
        I_exact_E_dir *= kweight
        I_exact_ortho *= kweight

    Iw_exact_E_dir = fftshift(fft(I_exact_E_dir*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Iw_exact_ortho = fftshift(fft(I_exact_ortho*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Int_exact_E_dir = (freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = (freq**2)*np.abs(Iw_exact_ortho)**2

    I_exact_name = 'Iexact_' + tail
    np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])
    # Save the parameters of the calculation
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(params.__dict__))

    if save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution=solution, paths=paths,
                 electric_field=electric_field(t), A_field=A_field, Zee_field=Zee_field)

def diff(x, y):
    '''
    Takes the derivative of y w.r.t. x
    '''
    if len(x) != len(y):
        raise ValueError('Vectors have different lengths')
    if len(y) == 1:
        return 0

    dx = np.gradient(x)
    dy = np.gradient(y)
    return dy/dx


def gaussian_envelope(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2.0/(2.0*alpha)**2)


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
    di_10mxf = dipole_B.Mxfjit[1][0]
    di_11mxf = dipole_B.Mxfjit[1][1]

    di_00myf = dipole_B.Myfjit[0][0]
    di_10myf = dipole_B.Myfjit[1][0]
    di_11myf = dipole_B.Myfjit[1][1]

    di_00mzf = dipole_B.Mzfjit[0][0]
    di_10mzf = dipole_B.Mzfjit[1][0]
    di_11mzf = dipole_B.Mzfjit[1][1]

    @njit
    def flength(t, y, kpath, dk, y0):

        # WARNING! THE LENGTH GAUGE ONLY WORKS WITH
        # TIME CONSTANT MAGNETIC FIELDS FOR NOW
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
            x[i] = 2*(wr*y[i+1]).imag + D*(y[m] - y[n]) \
                - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr_c*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(wr*y[i+1]).imag + D*(y[m+3] - y[n+3]) \
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
            x[i] = 2*(wr*y[i+1]).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr_c*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(wr*y[i+1]).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    @njit
    def fvelocity_extra(t, y, kpath, dk, y0):

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

        # Electric dipoles
        di_00x = di_00xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01x = di_01xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11x = di_11xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_00y = di_00yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01y = di_01yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11y = di_11yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        # Magnetic dipole integral
        di_00mx = di_00mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_10mx = di_10mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11mx = di_11mxf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        di_00my = di_00myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_10my = di_10myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11my = di_11myf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        di_00mz = di_00mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_10mz = di_10mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11mz = di_11mzf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        # Electric
        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Electric field strength
        electric_f = electric_field(t)

        # Time derivative zeeman field
        m_zee_deriv = zeeman_field_derivative(t)
        mxd = m_zee_deriv[0]
        myd = m_zee_deriv[1]
        mzd = m_zee_deriv[2]

        # Magnetic
        M_in_path = mxd*di_10mx + myd*di_10my + mzd*di_10mz
        M_diag_in_path = mxd*(di_00mx - di_11mx) + myd*(di_00my - di_11my) \
            + mzd*(di_00mz - di_11mz)

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

            # Magnetic frequency: M_21(k).Bdot(t)
            mr_d_diag = M_diag_in_path[k]
            mr = M_in_path[k]
            mr_c = mr.conjugate()

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*((wr+mr)*y[i+1]).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*(wr_d_diag + mr_d_diag))*y[i+1] \
                - 1j*(wr_c + mr_c)*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*((wr+mr)*y[i+1]).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    freturn = None
    if (gauge == 'velocity'):
        print("Using velocity gauge")
        freturn = fvelocity
    if (gauge == 'length'):
        print("Using length gauge")
        freturn = flength
    if (gauge == "velocity_extra"):
        print("Using velocity extra gauge")
        freturn = fvelocity_extra

    def f(t, y, kpath, dk, y0):
        return freturn(t, y, kpath, dk, y0)

    return f

def initial_condition(e_fermi, temperature, e_c):
    knum = e_c.size
    ones = np.ones(knum, dtype=np.float64)
    zeros = np.zeros(knum, dtype=np.float64)
    distrib = np.zeros(knum, dtype=np.float64)
    if temperature > 1e-5:
        distrib += 1/(np.exp((e_c-e_fermi)/temperature) + 1)
        return np.array([ones, zeros, zeros, distrib]).flatten('F')

    smaller_e_fermi = (e_fermi - e_c) > 0
    distrib[smaller_e_fermi] += 1
    return np.array([ones, zeros, zeros, distrib]).flatten('F')


def BZ_plot(kpnts, a, b1, b2, paths, si_units=True):

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


def print_user_info(BZ_type, Nk, align, angle_inc_E_field, E0, B0, w, alpha, chirp,
                    T2, tfmt0, dt):

    print("Input parameters:")
    print("Brillouin zone:                 " + BZ_type)
    print("Number of k-points              = " + str(Nk))
    if BZ_type == 'full':
        print("Driving field alignment         = " + align)
    elif BZ_type == '2line':
        print("Driving field direction         = " + str(angle_inc_E_field))
    print("Driving amplitude (MV/cm)[a.u.] = " + "("
          + '{:.6f}'.format(E0*co.au_to_MVpcm) + ")"
          + "[" + '{:.6f}'.format(E0) + "]")
    print("Magnetic amplitude (T)[a.u.]    = " + "("
          + '%.6f'%(B0*co.au_to_T) + ")"
          + "[" + '%.6f'%(B0) + "]")
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
