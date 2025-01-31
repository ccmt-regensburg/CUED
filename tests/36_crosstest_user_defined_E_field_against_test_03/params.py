# Input parameters for SBE.py
import numpy as np


class params:
    # System parameters
    #########################################################################
    e_fermi           = 0.0                      # Fermi energy in eV
    temperature       = 0.0                      # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    BZ_type           = 'rectangle'              # rectangle or hexagon
    Nk1               = 50                       # Number of kpoints in each of the paths
    Nk2               = 4                        # Number of paths
    length_BZ_E_dir   = 5.0                      # length of BZ in E-field direction
    length_BZ_ortho   = 0.2                      # length of BZ orthogonal to E-field direction
    angle_inc_E_field = 0                        # incoming angle of the E-field in degree
    dk_order          = 8                        # order for numerical derivative of density matrix

    # Driving field parameters
    ##########################################################################
    f                 = 25.0                     # Pulse frequency (THz)

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1                = 1000                     # Phenomenological diagonal damping time
    T2                = 1                        # Phenomenological polarization damping time
    t0                = -1000                    # Start time *pulse centered @ t=0, use t0 << 0
    dt                = 0.05                     # Time step

    # Flags for testing and features
    ##########################################################################
    gauge                   = 'length'           # Gauge of the system
    solver                  = '2band'
    fourier_window_function = 'gaussian'
    gaussian_window_width   = 50
    user_out                = False
    save_latex_pdf          = False