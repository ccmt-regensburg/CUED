import numpy as np
from numpy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as pl
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

from hfsbe.brillouin import hex_mesh, mesh
from hfsbe.solver import make_electric_field, make_zeeman_field
from hfsbe.solver import emission_exact_zeeman
from hfsbe.solver import make_velocity_zeeman_solver as mvzs

fs_to_au = 41.34137335                   # (1fs = 41.341473335 a.u.)
MVpcm_to_au = 0.0001944690381            # (1MV/cm = 1.944690381*10^-4 a.u.)
T_to_au = 4.255e-6                       # (1T     = 4.255e-6 a.u.)
THz_to_au = 0.000024188843266            # (1THz   = 2.4188843266*10^-5 a.u.)
A_to_au = 150.97488474                   # (1A     = 150.97488474)
eV_to_au = 0.03674932176                 # (1eV    = 0.036749322176 a.u.)
mub_to_au = 0.5                          # (1mu_b    = 0.5 a.u. Bohr magneton)


def sbe_zeeman_solver(sys, dipole_k, dipole_B, params):

    # System parameters
    a = params.a                               # Lattice spacing
    e_fermi = params.e_fermi*eV_to_au          # Fermi energy
    temperature = params.temperature*eV_to_au  # Temperature

    # Driving field parameters
    E0 = params.E0*MVpcm_to_au                 # Driving pulse field amplitude
    w = params.w*THz_to_au                     # Driving pulse frequency
    chirp = params.chirp*THz_to_au             # Pulse chirp frequency
    alpha = params.alpha*fs_to_au              # Gaussian pulse width
    phase = params.phase                       # Carrier-envelope phase

    B0 = params.B0*T_to_au
    incident_angle = np.radians(params.incident_angle)
    mu = mub_to_au*np.array([params.mu_x, params.mu_y, params.mu_z])

    # Time scales
    T1 = params.T1*fs_to_au                    # Occupation damping time
    T2 = params.T2*fs_to_au                    # Polarization damping time
    gamma1 = 1/T1                              # Occupation damping parameter
    gamma2 = 1/T2                              # Polarization damping param
    t0 = int(params.t0*fs_to_au)               # Initial time condition
    tf = int(params.tf*fs_to_au)               # Final time
    dt = params.dt*fs_to_au                    # Integration time step
    dt_out = 1/(2*params.dt)                   # Solution output time step

    # Brillouin zone type
    BZ_type = params.BZ_type                   # Type of Brillouin zone

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1 = params.Nk1                       # kpoints in b1 direction
        Nk2 = params.Nk2                       # kpoints in b2 direction
        Nk = Nk1*Nk2                           # Total number of kpoints
        align = params.align                   # E-field alignment
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path
        Nk = 2*Nk_in_path
        # rel_dist_to_Gamma = params.rel_dist_to_Gamma
        # length_path_in_BZ = params.length_path_in_BZ
        angle_inc_E_field = params.angle_inc_E_field
        align = str(angle_inc_E_field)

    b1 = params.b1                             # Reciprocal lattice vectors
    b2 = params.b2

    print_info(BZ_type, Nk, align, E0, B0, w, alpha, chirp, T2, tf-t0, dt)
    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        kpnts, paths = hex_mesh(Nk1, Nk2, a, b1, b2, align)
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
        BZ_plot(kpnts, a, b1, b2, paths)

    # Number of integration steps, time array construction flag
    Nt = int((tf-t0)/dt)
    t_constructed = False

    # Solution containers
    t = []
    solution = []

    # Initialize the ode solver and create fnumba
    electric_field = make_electric_field(E0, w, alpha, chirp, phase)
    zeeman_field = make_zeeman_field(B0, mu, w, alpha, chirp, phase, E_dir,
                                     incident_angle)

    fnumba = mvzs(sys, dipole_k, dipole_B, gamma1, gamma2, E_dir,
                  electric_field, zeeman_field)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method='bdf', max_step=dt)

    # Vector field
    A_field = []
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone

    path_num = 1
    for path in paths:

        print('path: ' + str(path_num))

        # Solution container for the current path
        path_solution = []

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
        ti = 0
        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti % 1000 == 0):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step
            if (ti % dt_out == 0):
                # Do not append the last element (A_field)
                path_solution.append(solver.y[:-1])
                # Construct time array only once
                if not t_constructed:
                    # Construct time and A_field only in first round
                    t.append(solver.t)
                    A_field.append(solver.y[-1])

            # Increment time counter
            ti += 1

        # Flag that time array has been built up
        t_constructed = True
        path_num += 1

        # Append path solutions to the total solution arrays
        solution.append(path_solution)
        print(np.shape(path_solution))

    # Convert solution and time array to numpy arrays
    t = np.array(t)
    solution = np.array(solution)
    A_field = np.array(A_field)

    # Slice solution along each path for easier observable calculation
    # Split the last index into 100 subarrays, corresponding to kx
    # Consquently the new last axis becomes 4.
    if BZ_type == 'full':
        solution = np.array_split(solution, Nk1, axis=2)
    elif BZ_type == '2line':
        solution = np.array_split(solution, Nk_in_path, axis=2)

    # Convert lists into numpy arrays
    solution = np.array(solution)
    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e

    solution = shift_solution(solution, A_field, dk)

    # COMPUTE OBSERVABLES
    ###########################################################################
    dt_out = t[1] - t[0]
    freq = fftshift(fftfreq(np.size(t), d=dt_out))

    I_exact_E_dir, I_exact_ortho =\
        emission_exact_zeeman(sys, paths, t, solution, E_dir, A_field,
                              zeeman_field)

    Iw_exact_E_dir = fftshift(fft(I_exact_E_dir*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Iw_exact_ortho = fftshift(fft(I_exact_ortho*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Int_exact_E_dir = (freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = (freq**2)*np.abs(Iw_exact_ortho)**2

    # Save observables to file
    if (BZ_type == '2line'):
        Nk1 = Nk_in_path
        Nk2 = 2

    tail = 'Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}'\
        .format(Nk1, Nk2, w/THz_to_au, E0/MVpcm_to_au, alpha/fs_to_au, phase, T2/fs_to_au)

    I_exact_name = 'Iexact_' + tail
    np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho, freq/w,
            Iw_exact_E_dir, Iw_exact_ortho,
            Int_exact_E_dir, Int_exact_ortho])


def gaussian_envelope(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2.0/(2.0*alpha)**2)


def shift_solution(solution, A_field, dk):

    for i_time in range(np.size(A_field)):
        # shift of k index in the direction of the E-field
        # (direction is already included in the paths)
        k_shift = (A_field[i_time]/dk).real
        k_index_shift_1 = int(int(np.abs(k_shift))*np.sign(k_shift))
        if (k_shift < 0):
            k_index_shift_1 = k_index_shift_1 - 1
        k_index_shift_2 = k_index_shift_1 + 1
        weight_1 = k_index_shift_2 - k_shift
        weight_2 = 1 - weight_1
        # n_kpoints = np.size(solution[:, 0, 0, 0])

        # transfer to polar coordinates
        r = np.abs(solution[:, :, i_time, :])
        phi = np.arctan2(np.imag(solution[:, :, i_time, :]),
                         np.real(solution[:, :, i_time, :]))

        r = weight_1*np.roll(r,   k_index_shift_1, axis=0) \
            + weight_2*np.roll(r,   k_index_shift_2, axis=0)
        phi = weight_1*np.roll(phi, k_index_shift_1, axis=0) \
            + weight_2*np.roll(phi, k_index_shift_2, axis=0)

        solution[:, :, i_time, :] = r*np.cos(phi) + 1j*r*np.sin(phi)

    return solution


def initial_condition(e_fermi, temperature, e_c):
    knum = e_c.size
    ones = np.ones(knum)
    zeros = np.zeros(knum)
    if (temperature > 1e-5):
        distrib = 1/(np.exp((e_c-e_fermi)/temperature)+1)
        return np.array([ones, zeros, zeros, distrib]).flatten('F')
    else:
        return np.array([ones, zeros, zeros, zeros]).flatten('F')


def BZ_plot(kpnts, a, b1, b2, paths):

    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = pl.figure(figsize=(10, 10))
    ax = BZ_fig.add_subplot(111, aspect='equal')

    for b in ((0, 0), b1, -b1, b2, -b2, b1+b2, -b1-b2):
        poly = RegularPolygon(b, 6, radius=R, orientation=np.pi/6, fill=False)
        ax.add_patch(poly)

#    ax.arrow(-0.5*E_dir[0], -0.5*E_dir[1], E_dir[0], E_dir[1],
#             width=0.005, alpha=0.5, label='E-field')

    pl.scatter(0, 0, s=15, c='black')
    pl.text(0.01, 0.01, r'$\Gamma$')
    pl.scatter(r*np.cos(-np.pi/6), r*np.sin(-np.pi/6), s=15, c='black')
    pl.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, r'$M$')
    pl.scatter(R, 0, s=15, c='black')
    pl.text(R, 0.02, r'$K$')
    pl.scatter(kpnts[:, 0], kpnts[:, 1], s=15)
    pl.xlim(-7.0/a, 7.0/a)
    pl.ylim(-7.0/a, 7.0/a)
    pl.xlabel(r'$k_x$ ($1/a_0$)')
    pl.ylabel(r'$k_y$ ($1/a_0$)')

    for path in paths:
        path = np.array(path)
        pl.plot(path[:, 0], path[:, 1])

    pl.show()


def print_info(BZ_type, Nk, align, E0, B0, w, alpha, chirp, T2, tft0, dt):
    # USER OUTPUT
    ###########################################################################
    print("Solving for...")
    print("Brillouin zone: " + BZ_type)
    print("Number of k-points              = " + str(Nk))
    print("Driving field alignment     = " + align)

    print("Driving amplitude (MV/cm)[a.u.] = "
          + "(" + '%.6f' % (E0/MVpcm_to_au) + ")" + "[" + '%.6f' % (E0) + "]")
    print("Magnetic  amplitude (T)[a.u]    = "
          + "(" + '%.6f' % (B0/T_to_au) + ")" + "[" + '%.6f' % (B0) + "]")
    print("Pulse Frequency (THz)[a.u.]     = "
          + "(" + '%.6f' % (w/THz_to_au) + ")" + "[" + '%.6f' % (w) + "]")
    print("Pulse Width (fs)[a.u.]          = "
          + "(" + '%.6f' % (alpha/fs_to_au) + ")" + "[" + '%.6f' % (alpha) + "]")
    print("Chirp rate (THz)[a.u.]          = "
          + "(" + '%.6f' % (chirp/THz_to_au) + ")" + "[" + '%.6f' % (chirp) + "]")
    print("Damping time (fs)[a.u.]         = "
          + "(" + '%.6f' % (T2/fs_to_au) + ")" + "[" + '%.6f' % (T2) + "]")
    print("Total time (fs)[a.u.]           = "
          + "(" + '%.6f' % ((tft0)/fs_to_au) + ")" + "[" + '%.5i' % (tft0) + "]")
    print("Time step (fs)[a.u.]            = "
          + "(" + '%.6f' % (dt/fs_to_au) + ")" + "[" + '%.6f' % (dt) + "]")

