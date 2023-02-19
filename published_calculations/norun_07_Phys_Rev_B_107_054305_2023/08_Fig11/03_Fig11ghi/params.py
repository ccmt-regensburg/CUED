# Input parameters for SBE.py
import numpy as np

# Variable for test_script.py
MPI_JOBS=8


class params:
        # System parameters
        #########################################################################
        e_fermi             = 0.00        # Fermi energy in eV
        temperature         = 0.03        # Temperature in eV

        # Model Hamiltonian parameters
        # Brillouin zone parameters
        ##########################################################################
        # Type of Brillouin zone
        BZ_type             = 'rectangle'            # rectangle or hexagon
        Nk1                 = 1200                     # Number of kpoints in each of the paths
        Nk2                 = 1                      # Number of paths
        length_BZ_ortho     = 1            #
        angle_inc_E_field   = 0                      # incoming angle of the E-field in degree

        # Driving field parameters
        ##########################################################################
        E0                  = 3         # Pulse amplitude (MV/cm)
        f                   = 40.0        # Pulse frequency (THz)
        chirp               = [-2,0,2]       # Pulse chirp ratio (chirp = c/w) (THz)
        sigma               = 50          # Gaussian pulse width (femtoseconds)
        phase               = np.linspace(0,2*np.pi,num=384)

        # Time scales (all units in femtoseconds)
        ##########################################################################
        T1    = 10     # Phenomenological diagonal damping time
        T2    = 1       # Phenomenological polarization damping time
        t0    = -600    # Start time *pulse centered @ t=0, use t0 << 0
        dt    = 0.005      # Time step

        # Flags for testing and features
        ##########################################################################
        gauge                   = 'velocity'   # Gauge of the system
        solver                  = 'nband'
        fourier_window_function = 'gaussian'
        user_out                = False
        parallelize_over_points = False
        plot_format             = 'png'
        save_latex_pdf          = False
