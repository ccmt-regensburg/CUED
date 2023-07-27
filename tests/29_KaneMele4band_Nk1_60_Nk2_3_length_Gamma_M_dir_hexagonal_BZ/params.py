# Input parameters for SBE.py
import numpy as np

MPI_NUM_PROCS=3

class params:
    # System parameters
    #########################################################################
    a                   = 1.890 * 5.16             # Lattice spacing in atomic units (5.16 A)
    e_fermi             = 0.0                    # Fermi energy in eV
    temperature         = 0.0                    # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    # 'hexagon' for full hexagonal BZ, 'rectangle' for rectangle with adjustable size
    BZ_type = 'hexagon'
    # path_type = 'KGM'

    # hexagonal BZ parameters
    Nk1                 = 60                 # Number of kpoints in each of the paths
    Nk2                 = 3                  # Number of paths

    # Driving field parameters
    ##########################################################################
    align               = 'M'
    E0                  = 3               # Pulse amplitude (MV/cm)
    f                   = 25.0                   # Pulse frequency (THz)
    chirp               = 0.00                   # Pulse chirp ratio (chirp = c/w) (THz)
    sigma               = 50.0                   # Gaussian pulse width (femtoseconds)
    phase               = 0.0

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000                                 # Phenomenological diagonal damping time
    T2    = 10                                    # Phenomenological polarization damping time
    t0    = -1000                                 # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.01                                 # Time step

    # Flags for testing and features
    ##########################################################################

    gauge                   = 'length'           # Gauge of the system
    solver                  = 'nband'
    solver_method           = 'bdf'
    fourier_window_function = 'gaussian'
    user_out                = False
    save_latex_pdf          = False
    path_parallelization    = True
    
