from math import modf
import sys
import numpy as np

from cued.utility import ConversionFactors as CoFa

class Params():
    """
    Environment variable class holding all relevant parameters in the SBE code.
    Additional check for ill-defined user parameters included.
    """
    def __init__(self, UP):
        # Command line progress output
        self.user_out = True
        if hasattr(UP, 'user_out'):
            self.user_out = UP.user_out

        # Save full density matrix
        self.save_full = False
        if hasattr(UP, 'save_full'):
            self.save_full = UP.save_full

        # Save data as human readable PDF file
        self.save_latex_pdf = False
        if hasattr(UP, 'save_latex_pdf'):
            self.save_latex_pdf = UP.save_latex_pdf

        # Save j^intra, j^anom, dP^inter/dt
        self.split_current = False
        if hasattr(UP, 'split_current'):
            self.split_current = UP.split_current

        # Semiclassical calc. (dipole = 0)
        self.do_semicl = False
        if hasattr(UP, 'do_semicl'):
            self.do_semicl = UP.do_semicl

        # Gauge of the SBE Dynamics
        self.gauge = 'length'
        if hasattr(UP, 'gauge'):
            self.gauge = UP.gauge

        # Numerical or analytical calculation of eigenstates and dipoles
        self.hamiltonian_evaluation = 'num'
        if hasattr(UP, 'hamiltonian_evaluation'):
            self.hamiltonian_evaluation = UP.hamiltonian_evaluation

        # 2 or n band solver
        self.solver = 'nband'
        if hasattr(UP, 'solver'):
            self.solver = UP.solver

        # Step size for num. derivatives
        self.epsilon = 2e-5
        if hasattr(UP, 'epsilon'):
            self.epsilon = UP.epsilon

        # Index of real wave function entry (fixes gauge)
        self.gidx = 1
        if hasattr(UP, 'gidx'):
            self.gidx = UP.gidx

        # Calculate anomalous current
        self.save_anom = False
        if hasattr(UP, 'save_anom'):
            self.save_anom = UP.save_anom

        # 'adams' non-stiff, 'bdf' stiff, 'rk4' Runge-Kutta 4th order
        self.solver_method = 'bdf'
        if hasattr(UP, 'solver_method'):
            self.solver_method = UP.solver_method

        # Quadruple for reducing numerical noise
        self.precision = 'double'
        if hasattr(UP, 'precision'):
            self.precision = UP.precision

        # Special flag for accurate insulator calc.
        self.symmetric_insulator = False
        if hasattr(UP, 'symmetric_insulator'):
            self.symmetric_insulator = UP.symmetric_insulator

        # Accuracy order of density-matrix k-deriv.
        self.dk_order = 8
        if hasattr(UP, 'dk_order'):
            self.dk_order = UP.dk_order                 # with length gauge (avail: 2,4,6,8)
            if self.dk_order not in [2, 4, 6, 8]:
                sys.exit("dk_order needs to be either 2, 4, 6, or 8.")

        # Parameters for initial occupation
        self.e_fermi = UP.e_fermi*CoFa.eV_to_au           # Fermi energy
        self.temperature = UP.temperature*CoFa.eV_to_au   # Temperature

        # Driving field parameters
        self.f = UP.f*CoFa.THz_to_au                      # Driving pulse frequency
        self.E0 = UP.E0*CoFa.MVpcm_to_au                  # Driving pulse field amplitude
        self.chirp = UP.chirp*CoFa.THz_to_au              # Pulse chirp frequency
        self.sigma = UP.sigma*CoFa.fs_to_au               # Gaussian pulse width
        self.phase = UP.phase                           # Carrier-envelope phase

        # Time scales
        self.T1 = UP.T1*CoFa.fs_to_au                     # Occupation damping time
        self.T2 = UP.T2*CoFa.fs_to_au                     # Polarization damping time
        self.t0 = UP.t0*CoFa.fs_to_au
        self.dt = UP.dt*CoFa.fs_to_au

        self.factor_freq_resolution = 1
        if hasattr(UP, 'factor_freq_resolution'):
            self.factor_freq_resolution = UP.factor_freq_resolution

        # gaussian or hann
        self.fourier_window_function = 'gaussian'
        if hasattr(UP, 'fourier_window_function'):
            self.fourier_window_function = UP.fourier_window_function

        self.gaussian_window_width = self.sigma
        if hasattr(UP, 'gaussian_window_width'):
            self.gaussian_window_width = UP.gaussian_window_width*CoFa.fs_to_au

        # Brillouin zone type
        self.BZ_type = UP.BZ_type                      # Type of Brillouin zone
        self.Nk1 = UP.Nk1                              # kpoints in b1 direction
        self.Nk2 = UP.Nk2                              # kpoints in b2 direction

        # Parameters for individual Brillouin zone types
        if self.BZ_type == 'hexagon':
            self.align = UP.align                      # E-field alignment
            self.angle_inc_E_field = None
            self.a = UP.a                              # Lattice spacing
        elif self.BZ_type == 'rectangle':
            self.align = None
            self.angle_inc_E_field = UP.angle_inc_E_field
            self.length_BZ_ortho = UP.length_BZ_ortho  # Size of the Brillouin zone in atomic units
            self.length_BZ_E_dir = UP.length_BZ_E_dir  # -"-
        else:
            sys.exit("BZ_type needs to be either hexagon or rectangle.")

        # Check if user params has any ill-defined parameters
        self.__check_user_paramas_for_wrong_arguments(UP)

        self.__append_derived_parameters(UP)

    def __check_user_paramas_for_wrong_arguments(self, UP):
        """
        Compare default paramaters with user parameters.
        If there are user paramters not defined in the parameters
        give a warning and halt the code.
        """
        default_params = self.__dict__.keys()
        user_params = UP.__dict__.keys() - {'__weakref__', '__doc__', '__dict__', '__module__'}
        diff_params = (default_params | user_params) - default_params

        if diff_params:
            print("Error: The following parameters have no effect inside the current run:")
            for param in diff_params:
                print(param, end=' ')

            print()
            sys.exit()

    def __append_derived_parameters(self, UP):
        ##################################################
        ## The following parameters are derived parameters
        ## and can not be set in the params.py file
        ##################################################

        # Derived precision parameters
        if self.precision == 'double':
            self.type_real_np    = np.float64
            self.type_complex_np = np.complex128
        elif self.precision == 'quadruple':
            self.type_real_np    = np.float128
            self.type_complex_np = np.complex256
            if self.solver_method != 'rk4':
                sys.exit("Error: Quadruple precision only works with Runge-Kutta 4 ODE solver.")
        else:
            sys.exit("Only default or quadruple precision available.")

        # Derived initial condition
        self.e_fermi_eV = UP.e_fermi
        self.temperature_eV =  UP.temperature

        # Derived driving field parameters
        self.E0_MVpcm = UP.E0
        self.f_THz = UP.f
        self.chirp_THz = UP.chirp
        self.sigma_fs = UP.sigma

        # Derived time scale parameters
        self.gamma1 = 1/self.T1
        self.T1_fs = UP.T1
        self.gamma1_dfs = 1/self.T1_fs

        self.gamma2 = 1/self.T2
        self.T2_fs = UP.T2
        self.gamma2_dfs = 1/self.T2_fs

        self.t0_fs = UP.t0
        self.tf = -self.t0
        self.tf_fs = -self.t0_fs

        self.dt = self.type_real_np(self.dt)
        self.dt_fs = self.type_real_np(self.dt)

        Nf = int((abs(2*self.t0_fs))/self.dt_fs)
        if modf((2*self.t0_fs/self.dt_fs))[0] > 1e-12:
            print("WARNING: The time window divided by dt is not an integer.")
        # Define a proper time window if Nt exists
        # +1 assures the inclusion of tf in the calculation
        self.Nt = Nf + 1

        # Derived BZ parameter
        self.Nk = self.Nk1 * self.Nk2
        if self.BZ_type == 'hexagon':
            self.a_angs = self.a*CoFa.au_to_as

        # Filename tail
        self.tail =\
            'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}_solver_{:s}_dk_order{}'\
            .format(self.E0_MVpcm, self.f_THz, self.sigma_fs, self.gauge, self.t0_fs, self.dt_fs,
                    self.Nk1, self.Nk2, self.T1_fs, self.T2_fs, self.chirp_THz, self.phase,
                    self.solver_method, self.dk_order)
