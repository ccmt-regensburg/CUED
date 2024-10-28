# Input parameters for SBE.py
import numpy as np

T2_list = np.logspace(0.1, 1, num=10, base=10)*0.1
T2_list = np.append(T2_list, np.logspace(0.1, 1, num=10, base=10))
T2_list = np.append(T2_list, np.logspace(0.1, 1, num=10, base=10)*10)
#T2_list = np.append(T2_list, np.logspace(0.1, 1, num=10, base=10)*10)

class params:
        # System parameters
        #########################################################################
        e_fermi           = 0.2                      # Fermi energy in eV
        temperature       = 0.0                      # Temperature in eV

        # Model Hamiltonian parameters
        # Brillouin zone parameters
        ##########################################################################
        BZ_type           = 'rectangle'              # rectangle or hexagon
        Nk1               = [1080]                      # Number of kpoints in each of the paths
        Nk2               = 6*72                       # Number of paths
        length_BZ_E_dir   = [1.2]                      # length of BZ in E-field direction
        length_BZ_ortho   = [0.6]                      # length of BZ orthogonal to E-field direction
        angle_inc_E_field = 0                        # incoming angle of the E-field in degree
        dk_order          = 8                        # order for numerical derivative of density matrix

        # Driving field parameters
        ##########################################################################
        E0                = [0.0012589254117941673]
        f                 = 0.1                     # Pulse frequency (THz)
        sigma             = 50.0
        chirp             = 0
        phase             = 0
        # Time scales (all units in femtoseconds)
        ##########################################################################
        T1                = 10                  # Phenomenological diagonal damping time
        T2                = T2_list                       # Phenomenological polarization damping time
        t0                = -500                     # Start time *pulse centered @ t=0, use t0 << 0
        dt                = [0.008]                     # Time step

        # Flags for testing and features
        ##########################################################################
        gauge                   = 'velocity'           # Gauge of the system
        solver                  = '2band'
        solver_method           = 'rk4'
        split_current                  = True
        user_defined_header     = 'Nk2=' + str(Nk2)
