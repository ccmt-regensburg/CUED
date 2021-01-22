# Input parameters for SBE.py
import numpy as np


class params:
    # System parameters
    #########################################################################
    a                   = 8.304       # Lattice spacing in atomic units (4.395 A)
    e_fermi             = 0.0         # Fermi energy in eV
    temperature         = 0.0         # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    BZ_type             = 'rectangle' # rectangle or hexagon
    Nk1                 = 30          # Number of kpoints in each of the paths
    Nk2                 = 2          # Number of paths
#    rel_dist_to_Gamma   = 0.05       # relative distance (in units of 2pi/a) of both paths to Gamma
#    length_path_in_BZ   = 1500*0.00306  # Length of path in BZ K-direction
    length_BZ_E_dir     = 1500*0.00306
    length_BZ_ortho     = 0.008*2*np.pi/8.304*4

    # Driving field parameters
    ##########################################################################
    E0                  = 10.0        # Pulse amplitude (MV/cm)
    w                   = 25.0        # Pulse frequency (THz)
    chirp               = 0.00        # Pulse chirp ratio (chirp = c/w) (THz)
    alpha               = 25.0        # Gaussian pulse width (femtoseconds)
    phase               = 0.0
    solver_method       = 'rk4'
    angle_inc_E_field   = 0          # incoming angle of the E-field in degree

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000     # Phenomenological diagonal damping time
    T2    = 1        # Phenomenological polarization damping time
    t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.01    # Time step

    # Flags for testing and features
    ##########################################################################
    gauge         = 'length'   # Gauge of the system
    do_semicl     = False      # Turn all dipoles to 0 and use Berry curvature in emission
    user_out      = True       # Set to True to get user plotting and progress output
    save_approx   = True
    save_full     = False
    save_txt      = False

