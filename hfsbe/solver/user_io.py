import numpy as np

fs_to_au = 41.34137335                   # (1fs = 41.341473335 a.u.)
MVpcm_to_au = 0.0001944690381            # (1MV/cm = 1.944690381*10^-4 a.u.)
T_to_au = 4.255e-6                       # (1T     = 4.255e-6 a.u.)
THz_to_au = 0.000024188843266            # (1THz   = 2.4188843266*10^-5 a.u.)
A_to_au = 150.97488474                   # (1A     = 150.97488474)
eV_to_au = 0.03674932176                 # (1eV    = 0.036749322176 a.u.)
mub_to_au = 0.5                          # (1mu_b    = 0.5 a.u. Bohr magneton)


def read_zeeman_params(params):
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

    return a, e_fermi, temperature, E0, w, chirp, alpha, phase, B0, \
        incident_angle, mu, T1, T2, gamma1, gamma2 t0, tf, dt, dt_out, BZ_type


def print_zeeman_info(BZ_type, Nk, align, E0, B0, w, alpha, chirp, T2, tft0, dt):
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

