# Input parameters for SBE.py
import numpy as np


class params:
    # System parameters
    #########################################################################
    a                   = 8.308       # Lattice spacing in atomic units (4.395 A)
    e_fermi             = 0.0         # Fermi energy in eV
    temperature         = 0.0         # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    # 'hexagon' for full hexagonal BZ, 'rectangle' for rectangle with adjustable size
    BZ_type = 'hexagon'

    # hexagonal BZ parameters
    Nk1                 = 42         # Number of kpoints in each of the paths
    Nk2                 = 12         # Number of paths

    # Driving field parameters
    ##########################################################################
    align               = 'K'
    E0                  = 3.00        # Pulse amplitude (MV/cm)
    f                   = 25.0        # Pulse frequency (THz)
    chirp               = 0.00        # Pulse chirp ratio (chirp = c/w) (THz)
    sigma               = np.sqrt(2)*25.0        # Gaussian pulse width (femtoseconds)
    phase               = 0.0

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000     # Phenomenological diagonal damping time
    T2    = 1        # Phenomenological polarization damping time
    t0    = -400     # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.1      # Time step

    # Flags for testing and features
    ##########################################################################

    gauge                   = 'length'   # Gauge of the system
    hamiltonian_evaluation  = 'ana'
    solver                  = '2band'
    user_out                = False
    split_current           = False
    save_text               = True
