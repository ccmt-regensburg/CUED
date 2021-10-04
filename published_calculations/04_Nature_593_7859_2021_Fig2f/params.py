# Input parameters for SBE.py
import numpy as np

MPI_JOBS=8

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
    Nk1                 = 450       # Number of kpoints in each of the paths
    Nk2                 = 42         # Number of paths

    # Driving field parameters
    ##########################################################################
    align               = 'K'         # E-field direction (gamma-'K' or gamma-'M')
    E0                  = 3.0         # Pulse amplitude (MV/cm)
    f                   = 40.0        # Pulse frequency (THz)
    chirp               = 0.90       # Pulse chirp ratio (chirp = c/w) (THz)
    sigma               = 50          # Gaussian pulse width (femtoseconds)
    phase               = np.linspace(0, 2*np.pi, 16)

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 10       # Phenomenological diagonal damping time
    T2    = 1        # Phenomenological polarization damping time
    t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 5e-2     # Time step

    # Flags for testing and features
    ##########################################################################
    gauge          = 'length'   # Gauge of the system
    solver         = '2band'
    user_out       = False
    plot_format    = 'png'
