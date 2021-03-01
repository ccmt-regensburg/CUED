# Input parameters for SBE.py
import numpy as np


class params:
    # System parameters
    #########################################################################
    e_fermi             = 0.0         # Fermi energy in eV
    temperature         = 0.0         # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    BZ_type             = 'rectangle' # rectangle or hexagon
    Nk1                 = 2           # Number of kpoints in each of the paths
    Nk2                 = 2           # Number of paths
    length_BZ_E_dir     = 5.0         # length of BZ in E-field direction
    length_BZ_ortho     = 0.1         # length of BZ orthogonal to E-field direction
    angle_inc_E_field   = 0           # incoming angle of the E-field in degree

    # Driving field parameters
    ##########################################################################
    E0                  = 5.0         # Pulse amplitude (MV/cm)
    f                   = 25.0        # Pulse frequency (THz)
    chirp               = 0.00        # Pulse chirp ratio (chirp = c/w) (THz)
    alpha               = 25.0        # Gaussian pulse width (femtoseconds)
    phase               = 0.0
    angle_inc_E_field   = 0           # incoming angle of the E-field in degree

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000     # Phenomenological diagonal damping time
    T2    = 1        # Phenomenological polarization damping time
    t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.05    # Time step

    # Flags for testing and features
    ##########################################################################
    gauge                   = 'velocity'   # Gauge of the system
    hamiltonian_evaluation  = 'ana'
    solver                  = '2band'
    do_semicl               = True         # Turn all dipoles to 0 and use Berry curvature in emission
    user_out                = False         # Set to True to get user plotting and progress output
    current_splitting       = True
    save_full               = False

