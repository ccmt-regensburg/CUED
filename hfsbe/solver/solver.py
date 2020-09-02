import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numba import njit
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

from hfsbe.utility import conversion_factors as co
from hfsbe.solver import polarization, current, make_emission_exact
from hfsbe.solver import make_electric_field


def sbe_solver(sys, dipole, params):
    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_file = params.save_file
    save_full = params.save_full
    save_approx = params.save_approx
    dipole_off = params.dipole_off
    gauge = params.gauge

    # System parameters
    a = params.a                                # Lattice spacing
    e_fermi = params.e_fermi*co.eV_to_au           # Fermi energy
    temperature = params.temperature*co.eV_to_au   # Temperature

    # Driving field parameters
    E0 = params.E0*co.MVpcm_to_au               # Driving pulse field amplitude
    w = params.w*co.THz_to_au                   # Driving pulse frequency
    chirp = params.chirp*co.THz_to_au              # Pulse chirp frequency
    alpha = params.alpha*co.fs_to_au               # Gaussian pulse width
    phase = params.phase                        # Carrier-envelope phase

    # Time scales
    T1 = params.T1*co.fs_to_au                     # Occupation damping time
    T2 = params.T2*co.fs_to_au                     # Polarization damping time
    gamma1 = 1/T1                               # Occupation damping parameter
    gamma2 = 1/T2                               # Polarization damping param

    Nf = int((abs(2*params.t0))/params.dt)
    # Find out integer times Nt fits into total time steps
    dt_out = int(ceil(Nf/params.Nt))

    # Expand time window to fit Nf output steps
    Nt = dt_out*params.Nt
    total_fs = Nt*params.dt
    t0 = (-total_fs/2)*co.fs_to_au
    tf = (total_fs/2)*co.fs_to_au
    dt = params.dt*co.fs_to_au

    # Brillouin zone type
    BZ_type = params.BZ_type                    # Type of Brillouin zone

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1 = params.Nk1                        # kpoints in b1 direction
        Nk2 = params.Nk2                        # kpoints in b2 direction
        Nk = Nk1*Nk2                            # Total number of kpoints
        align = params.align                    # E-field alignment
        angle_inc_E_field = None
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path
        Nk = 2*Nk_in_path
        align = None
        angle_inc_E_field = params.angle_inc_E_field
        Nk1 = Nk_in_path
        Nk2 = 2

    b1 = params.b1                              # Reciprocal lattice vectors
    b2 = params.b2

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print_user_info(BZ_type, Nk, align, angle_inc_E_field, E0, w, alpha,
                        chirp, T2, tf-t0, dt)
    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        kpnts, paths = hex_mesh(Nk1, Nk2, a, b1, b2, align)
        dkx = np.abs(paths[0, 0] - paths[0, 1])[0]
        dky = np.abs(paths[0, 0] - paths[1, 0])[1]
        dk = 1/Nk1
        if align == 'K':
            E_dir = np.array([1, 0])
        elif align == 'M':
            E_dir = np.array([np.cos(np.radians(-30)),
                             np.sin(np.radians(-30))])
        BZ_plot(kpnts, a, b1, b2, paths)
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                         np.sin(np.radians(angle_inc_E_field))])
        dk, kpnts, paths = mesh(params, E_dir)
        # BZ_plot(kpnts, a, b1, b2, paths)

    t_constructed = False
    # Solution containers
    t = np.empty(params.Nt)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    solution = np.empty((Nk1, Nk2, params.Nt, 4), dtype=complex)
    A_field = np.empty(params.Nt, dtype=complex)

    # Initalise electric_field, create fnumba and initalise ode solver
    electric_field = make_electric_field(E0, w, alpha, chirp, phase)
    fnumba = make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field,
                         gauge=gauge, dipole_off=dipole_off)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method='bdf', max_step=dt)

    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    for Nk2_idx, path in enumerate(paths):
        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        if (dipole_off):
            zeros = np.zeros(np.size(kx_in_path), dtype=np.complex)
            dipole_in_path = zeros
            A_in_path = zeros
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
            # A[0,1,:] means 0-1 offdiagonal element
            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # in bite.evaluate, there is also an interpolation done if b1, b2 are
        # provided and a cutoff radius
        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)
        ecv_in_path = ec - sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = initial_condition(e_fermi, temperature, ec)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0, t0)\
            .set_f_params(path, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

        # Propagate through time

        # Index of current integration time step
        ti = 0
        # Index of current output time step
        t_idx = 0

        while solver.successful() and ti < Nt:
            # User output of integration progress
            # print(ti)
            if (ti % 1000 == 0 and user_out):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step
            if (ti % dt_out == 0):
                # Do not append the last element (A_field)

                solution[:, Nk2_idx, t_idx, :] = solver.y[:-1].reshape(Nk1, 4)
                # Construct time array only once
                if not t_constructed:
                    # Construct time and A_field only in first round
                    t[t_idx] = solver.t
                    A_field[t_idx] = solver.y[-1]

                t_idx += 1
            # Increment time counter
            ti += 1

        # Flag that time array has been built up
        t_constructed = True

    # COMPUTE OBSERVABLES
    ###########################################################################
    # Filename tail
    tail = 'Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}'\
        .format(Nk1, Nk2, w*co.au_to_THz, E0*co.au_to_MVpcm, alpha*co.au_to_fs,
                phase, T2*co.au_to_fs)

    # Fourier transforms
    dt_out = t[1] - t[0]
    freq = fftshift(fftfreq(np.size(t), d=dt_out))
    if (gauge == 'length' and save_approx == True):
        # Only calculate kira & koch emission if we are in length gauge
        # Calculate parallel and orthogonal components of observables
        # Polarization (interband)
        P_E_dir, P_ortho = polarization(dipole, paths, solution[:, :, :, 1], E_dir)
        # Current (intraband)
        J_E_dir, J_ortho = current(sys, paths, solution[:, :, :, 0],
                                   solution[:, :, :, 3], t, alpha, E_dir)
        # Approximate emission in time
        I_E_dir = diff(t, P_E_dir)*gaussian_envelope(t, alpha) \
            + J_E_dir*gaussian_envelope(t, alpha)
        I_ortho = diff(t, P_ortho)*gaussian_envelope(t, alpha) \
            + J_ortho*gaussian_envelope(t, alpha)
        if (BZ_type == '2line'):
            I_E_dir *= (dk/(4*np.pi))
            I_ortho *= (dk/(4*np.pi))
        if (BZ_type == 'full'):
            I_E_dir *= (dkx*dky/(2*np.pi)**2)
            I_ortho *= (dkx*dky/(2*np.pi)**2)


        Iw_E_dir = fftshift(fft(I_E_dir, norm='ortho'))
        Iw_ortho = fftshift(fft(I_ortho, norm='ortho'))
        # Pw_E_dir = fftshift(fft(diff(t, P_E_dir), norm='ortho'))
        # Pw_ortho = fftshift(fft(diff(t, P_ortho), norm='ortho'))
        # # Jw_E_dir = fftshift(fft(J_E_dir*gaussian_envelope(t, alpha), norm='ortho'))
        # Jw_ortho = fftshift(fft(J_ortho*gaussian_envelope(t, alpha), norm='ortho'))

        # Approximate Emission intensity
        Int_E_dir = (freq**2)*np.abs(Iw_E_dir)**2
        Int_ortho = (freq**2)*np.abs(Iw_ortho)**2

        I_approx_name = 'Iapprox_' + tail
        np.save(I_approx_name, [t, I_E_dir, I_ortho,
                freq/w, Iw_E_dir, Iw_ortho, Int_E_dir, Int_ortho])

    ##############################################################
    # Always calculate exact emission formula
    ##############################################################
    emission_exact = make_emission_exact(sys, paths, solution,
                                         E_dir, A_field, gauge)
    I_exact_E_dir, I_exact_ortho = emission_exact()
    if (BZ_type == '2line'):
        I_exact_E_dir *= (dk/(4*np.pi))
        I_exact_ortho *= (dk/(4*np.pi))
    if (BZ_type == 'full'):
        I_exact_E_dir *= (dkx*dky/(2*np.pi)**2)
        I_exact_ortho *= (dkx*dky/(2*np.pi)**2)

    Iw_exact_E_dir = fftshift(fft(I_exact_E_dir*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Iw_exact_ortho = fftshift(fft(I_exact_ortho*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Int_exact_E_dir = (freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = (freq**2)*np.abs(Iw_exact_ortho)**2

    # Save observables to file
    if (save_file):
        I_exact_name = 'Iexact_' + tail
        np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho, freq/w,
                Iw_exact_E_dir, Iw_exact_ortho,
                Int_exact_E_dir, Int_exact_ortho])
        # Save the parameters of the calculation
        params_name = 'params.txt'
        paramsfile = open(params_name, 'w')
        paramsfile.write(str(params.__dict__))

    if (save_full):
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution=solution, paths=paths,
                 electric_field=electric_field(t), A_field=A_field)

    # J_name = 'J_' + tail
    # np.save(J_name, [t, J_E_dir, J_ortho, freq/w, Jw_E_dir, Jw_ortho])
    # P_name = 'P_' + tail
    # np.save(P_name, [t, P_E_dir, P_ortho, freq/w, Pw_E_dir, Pw_ortho])
    # I_name = 'I_' + tail
    # np.save(I_name, [t, I_E_dir, I_ortho, freq/w, np.abs(Iw_E_dir),
    #                  np.abs(Iw_ortho), Int_E_dir, Int_ortho])

    # driving_tail = 'w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_wc-{:4.3f}'\
    #     .format(w/THz_conv, E0/E_conv, alpha/fs_conv, phase, chirp/THz_conv)

    # D_name = 'E_' + driving_tail
    # np.save(D_name, [t, electric_field(t)])


###############################################################################
# FUNCTIONS
###############################################################################
def mesh(params, E_dir):
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)),
                              0.5 - (1/(2*Nk_in_path)), num=Nk_in_path)
    vec_k_path = E_dir*length_path_in_BZ

    vec_k_ortho = 2.0*(np.pi/a)*rel_dist_to_Gamma*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
    for path_index in [-1, 1]:
        # Container for a single path
        path = []
        for alpha in alpha_array:
            # Create a k-point
            kpoint = path_index*vec_k_ortho + alpha*vec_k_path

            mesh.append(kpoint)
            path.append(kpoint)

        # Append the a1'th path to the paths array
        paths.append(path)

    dk = length_path_in_BZ/(Nk_in_path)

    return dk, np.array(mesh), np.array(paths)


def hex_mesh(Nk1, Nk2, a, b1, b2, align):
    alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num=Nk1)
    alpha2 = np.linspace(-0.5 + (1/(2*Nk2)), 0.5 - (1/(2*Nk2)), num=Nk2)

    def is_in_hex(p, a):
        # Returns true if the point is in the hexagonal BZ.
        # Checks if the absolute values of x and y components of p are within
        # the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*a)) and
                (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*a)))

    def reflect_point(p, a, b1, b2):
        x = p[0]
        y = p[1]
        if (y > 2*np.pi/(np.sqrt(3)*a)):   # Crosses top
            p -= b2
        elif (y < -2*np.pi/(np.sqrt(3)*a)):
            # Crosses bottom
            p += b2
        elif (np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):
            # Crosses top-right
            p -= b1 + b2
        elif (-np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):
            # Crosses bot-right
            p -= b1
        elif (np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):
            # Crosses bot-left
            p += b1 + b2
        elif (-np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):
            # Crosses top-left
            p += b1
        return p

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the Monkhorst-Pack mesh
    if align == 'M':
        for a2 in alpha2:
            # Container for a single gamma-M path
            path_M = []
            for a1 in alpha1:
                # Create a k-point
                kpoint = a1*b1 + a2*b2
                # If current point is in BZ, append it to the mesh and path_M
                if (is_in_hex(kpoint, a)):
                    mesh.append(kpoint)
                    path_M.append(kpoint)
                # If current point is NOT in BZ, reflect it along
                # the appropriate axis to get it in the BZ, then append.
                else:
                    while (not is_in_hex(kpoint, a)):
                        kpoint = reflect_point(kpoint, a, b1, b2)
                    mesh.append(kpoint)
                    path_M.append(kpoint)
            # Append the a1'th path to the paths array
            paths.append(path_M)

    elif align == 'K_jack':
        b_a1 = 8*np.pi/(a*3)*np.array([1, 0])
        b_a2 = 4*np.pi/(a*3)*np.array([1, np.sqrt(3)])
        # Extend over half of the b2 direction and 1.5x the b1 direction
        # (extending into the 2nd BZ to get correct boundary conditions)
        alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 1.0 - (1/(2*Nk1)), Nk1)
        alpha2 = np.linspace(0, 0.5 - (1/(2*Nk2)), Nk2)
        for a2 in alpha2:
            path_K = []
            for a1 in alpha1:
                kpoint = a1*b_a1 + a2*b_a2
                if is_in_hex(kpoint, a):
                    mesh.append(kpoint)
                    path_K.append(kpoint)
                else:
                    kpoint -= (2*np.pi/a) * np.array([1, 1/np.sqrt(3)])
                    mesh.append(kpoint)
                    path_K.append(kpoint)
            paths.append(path_K)

    elif align == 'K':
        if (Nk1%3 != 0 or Nk1%2 != 0):
            raise RuntimeError("Nk1: " + "{:.d}".format(Nk1) +
                               "needs to be divisible by 3 and even")
        elif (Nk2%3 != 0):
            raise RuntimeError("Nk2: " + "{:.d}".format(Nk2) +
                               "needs to be divisible by 3")
        b_a1 = 8*np.pi/(a*3)*np.array([1, 0])
        b_a2 = 4*np.pi/(a*3)*np.array([0, np.sqrt(3)])
        # Extend over half of the b2 direction and 1.5x the b1 direction
        # (extending into the 2nd BZ to get correct boundary conditions)
        alpha1 = np.linspace(-0.5 + (1.5/(2*Nk1)), 1.0 - (1.5/(2*Nk1)), Nk1)
        alpha2 = np.linspace(0, 0.5 - (1/(2*Nk2)), Nk2)
        for a2 in alpha2:
            path_K = []
            for a1 in alpha1:
                kpoint = a1*b_a1 + a2*b_a2
                if is_in_hex(kpoint, a):
                    mesh.append(kpoint)
                    path_K.append(kpoint)
                else:
                    while (not is_in_hex(kpoint, a)):
                        kpoint = reflect_point(kpoint, a, b1, b2)
                    # kpoint -= (2*np.pi/a) * np.array([1, 1/np.sqrt(3)])
                    mesh.append(kpoint)
                    path_K.append(kpoint)
            paths.append(path_K)

    return np.array(mesh), np.array(paths)


def diff(x, y):
    '''
    Takes the derivative of y w.r.t. x
    '''
    if (len(x) != len(y)):
        raise ValueError('Vectors have different lengths')
    elif len(y) == 1:
        return 0
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)
        return dy/dx


def gaussian_envelope(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    # sigma = sqrt(2)*alpha
    # # 1/(2*np.sqrt(np.pi)*alpha)*np.exp(-t**2/(2*alpha)**2)
    return np.exp(-t**2/(2*alpha)**2) 


def make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field, gauge,
                dipole_off):

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
    def flength(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

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
    def fvelocity(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Velocity gauge needs a recalculation of energies and dipoles sind k
        is shifted according to the vector potential A
        """

        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        k_shift = y[-1].real
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if (dipole_off):
            zeros = np.zeros(kx.size, dtype=np.complex128)
            dipole_in_path = zeros
            A_in_path = zeros
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

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

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

    freturn = None
    if (gauge == 'length'):
        print("Using length gauge")
        freturn = flength
    elif (gauge == 'velocity'):
        print("Using velocity gauge")
        freturn = fvelocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")

    def f(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        return freturn(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

    return f


def initial_condition(e_fermi, temperature, e_c):
    knum = e_c.size
    ones = np.ones(knum, dtype=np.float64)
    zeros = np.zeros(knum, dtype=np.float64)
    distrib = np.zeros(knum, dtype=np.float64)
    if (temperature > 1e-5):
        distrib += 1/(np.exp((e_c-e_fermi)/temperature) + 1)
        return np.array([ones, zeros, zeros, distrib]).flatten('F')
    else:
        smaller_e_fermi = (e_fermi - e_c) > 0
        distrib[smaller_e_fermi] += 1
        return np.array([ones, zeros, zeros, distrib]).flatten('F')


def BZ_plot(kpnts, a, b1, b2, paths, si_units=True):

    if (si_units):
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

    if (si_units):
        plt.xlabel(r'$k_x \text{ in } 1/\si{\angstrom}$')
        plt.ylabel(r'$k_y \text{ in } 1/\si{\angstrom}$')
    else:
        plt.xlabel(r'$k_x \text{ in } 1/a_0$')
        plt.ylabel(r'$k_y \text{ in } 1/a_0$')
    for path in paths:
        if (si_units):
            plt.plot(co.as_to_au*path[:, 0], co.as_to_au*path[:, 1])
        else:
            plt.plot(path[:, 0], path[:, 1])

    plt.show()


def print_user_info(BZ_type, Nk, align, angle_inc_E_field, E0, w, alpha, chirp,
                    T2, tfmt0, dt):

    print("Solving for...")
    print("Brillouin zone: " + BZ_type)
    print("Number of k-points              = " + str(Nk))
    if BZ_type == 'full':
        print("Driving field alignment         = " + align)
    elif BZ_type == '2line':
        print("Driving field direction         = " + str(angle_inc_E_field))
    print("Driving amplitude (MV/cm)[a.u.] = " + "("
          + '{:.6f}'.format(E0*co.au_to_MVpcm) + ")"
          + "[" + '{:.6f}'.format(E0) + "]")
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
