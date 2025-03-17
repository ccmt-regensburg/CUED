# Input parameters for SBE.py
import numpy as np

class params:
    # System parameters
    #########################################################################
    a                   = 8.308       # Lattice spacing in atomic units (4.395 A)
    e_fermi             = 0.00        # Fermi energy in eV
    temperature         = 0.03        # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    # 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
    BZ_type = 'hexagon'

    # full BZ parametes
    Nk1                 = 900         # Number of kpoints in each of the paths
    Nk2                 = 90          # Number of paths

    # Driving field parameters
    ##########################################################################
    align               = 'M'         # E-field direction (gamma-'K' or gamma-'M')
    E0                  = 3.0         # Pulse amplitude (MV/cm)
    f                   = 25.0        # Pulse frequency (THz)
    chirp               = -1.25       # Pulse chirp ratio (chirp = c/w) (THz)
    sigma               = 90          # Gaussian pulse width (femtoseconds)
    phase               = [0,np.pi/2]

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000     # Phenomenological diagonal damping time
    T2    = 10       # Phenomenological polarization damping time
    t0    = -500    # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.1      # Time step
    factor_freq_resolution = 10

    # Flags for testing and features
    ##########################################################################
    gauge                   = 'velocity'   # Gauge of the system
    solver                  = '2band'
    fourier_window_function = 'gaussian'
    user_out                = True
    plot_format             = 'png'
    
