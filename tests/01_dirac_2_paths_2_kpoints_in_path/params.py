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
    # 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
    BZ_type = '2line'

    # Reciprocal lattice vectors
    b1 = (2*np.pi/(a*np.sqrt(3)))*np.array([np.sqrt(3), -1])
    b2 = (4*np.pi/(a*np.sqrt(3)))*np.array([0, 1])

    # full BZ parametes
#    Nk1                 = 90         # Number of kpoints in b1 direction
#    Nk2                 = 12         # Number of kpoints in b2 direction (number of paths)

    # 2line BZ parameters
    # for Fig. 1b in Paper one has to set Nk1 = 1200 and Nk2 = number of paths
    Nk1                 = 2          # Number of kpoints in each of the paths
    Nk2                 = 2          # Number of paths

    rel_dist_to_Gamma   = 0.008        # relative distance (in units of 2pi/a) of both paths to Gamma
    length_path_in_BZ   = 1500*0.00306   # Length of path in BZ K-direction
    # length_path_in_BZ   = 4*np.pi/(np.sqrt(3)*a) # Length of path in BZ M-direction
    angle_inc_E_field   = 0           # incoming angle of the E-field in degree

    # Driving field parameters
    ##########################################################################
    align               = 'K'         # E-field direction (gamma-'K' or gamma-'M')
    E0                  = 5.00        # Pulse amplitude (MV/cm)
    w                   = 25.0        # Pulse frequency (THz)
    chirp               = 0.00        # Pulse chirp ratio (chirp = c/w) (THz)
    alpha               = 25.0        # Gaussian pulse width (femtoseconds)
    phase               = 0.0

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1000     # Phenomenological diagonal damping time
    T2    = 1        # Phenomenological polarization damping time
    t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
    dt    = 0.005    # Time step
    Nt    = 8192     # Length of result time array

    # Flags for testing and features
    ##########################################################################
    gauge         = 'velocity'   # Gauge of the system
    do_semicl     = False      # Turn all dipoles to 0 and use Berry curvature in emission
    user_out      = True       # Set to True to get user plotting and progress output
    save_approx   = True
    save_full     = False

