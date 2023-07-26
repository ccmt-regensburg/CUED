from math import modf
import sys
import numpy as np
import itertools

from cued.utility import ConversionFactors as CoFa

class ParamsParser():
    """
    Environment variable class holding all relevant parameters in the SBE code.
    Additional check for ill-defined user parameters included.
    """
    def __init__(self, UP):

        self.__combine_parameters(UP)

    def __combine_parameters(self, UP):

        self.parallelize_over_points = None
        self.split_paths = None
        self.split_order = 1

        # build dictionary of all parameters, exclude t_pdf_densmat, points_to_path and parameters of Gabor transformation
        excl_set = {'__weakref__', '__doc__', '__dict__', '__module__',"t_pdf_densmat","parallelize_over_points",'gabor_gaussian_center','gabor_window_width'}
        
        self.user_params = sorted(UP.__dict__.keys() - excl_set)
        self.t_pdf_densmat = np.array([-100, 0, 50, 100])*CoFa.fs_to_au   # Time points for printing density matrix
        if hasattr(UP, 'parallelize_over_points'):
            self.parallelize_over_points = UP.parallelize_over_points
        if hasattr(UP, 'split_paths'):
            self.split_paths = UP.split_paths
        if hasattr(UP, 't_pdf_densmat'):
            self.t_pdf_densmat = np.array(UP.t_pdf_densmat)*CoFa.fs_to_au # Time points for printing density matrix
        if hasattr(UP,'gabor_gaussian_center'):
            self.gabor_gaussian_center = np.array(UP.gabor_gaussian_center)*CoFa.fs_to_au
        if hasattr(UP,'gabor_window_width'):
            self.gabor_window_width = np.array(UP.gabor_window_width)*CoFa.fs_to_au
            
            
        # build list of parameter lists
        self.number_of_combinations = 1
        self.params_lists = []
        self.params_maximum = []

        for key in self.user_params:
            self.__append_to_list(UP.__dict__[key])

        # check, wheter Nk2 is given as a list
        self.path_list = False
        if type(UP.__dict__['Nk2']) == list or type(UP.__dict__['Nk2']) ==  np.ndarray:
            self.path_list = True

        # Build list with all possible parameter combinations
        self.params_combinations = list(itertools.product(*self.params_lists))


    def __append_to_list(self, param):
        if type(param) == list or type(param) == np.ndarray:
            if type(param) == np.ndarray:
                param = param.tolist()
            if not type(param[0]) == str:
                self.params_maximum.append(np.amax(param))
            else:
                self.params_maximum.append(0)
            self.number_of_combinations *= np.size(param)
            self.params_lists.append(param)
        else:
            self.params_lists.append([param])
            self.params_maximum.append(0)

    def construct_current_parameters_and_header(self, param_idx, UP):
        """
        Depending on the parameter indices (could be local when using MPI)
        split lists in the params class into individual parameter sets represented by
        a dictionary. Also automatically set the name of the output file on that parameter.
        Parameters
        ----------
        param_idx : int
            Index of the parameter to construct a dictionary from
        UP : class
            parameters of the params.py file
        """

        current_parameters = {}
        self.header = ''

        for key_idx, key in enumerate(self.user_params):
            current_parameters[key] = self.params_combinations[param_idx][key_idx]
            string_length = len('{:.4f}'.format(self.params_maximum[key_idx]))
            if type(UP.__dict__[key]) == list or type(UP.__dict__[key]) == np.ndarray:
                if type(current_parameters[key]) == str:
                    self.header += key + '=' + current_parameters[key] + '_'
                else:
                    self.header += key + '=' + ('{:.4f}'.format(current_parameters[key])).zfill(string_length) + '_'

        return current_parameters


    def distribute_parameters(self, param_idx, UP): # Take index of parameter (MPI-parallelized) and write current_parameters

        current_parameters = self.construct_current_parameters_and_header(param_idx, UP)

        self.__occupation(current_parameters)
        self.__time_scales(current_parameters)
        self.__field(current_parameters)
        self.__brillouin_zone(current_parameters)
        self.__optional(current_parameters)

        # Check if user params has any ill-defined parameters
        self.__check_user_params_for_wrong_arguments(current_parameters)
        self.__check_for_conflicting_parameter_cominations(current_parameters)
        self.__append_derived_parameters(current_parameters)


    def __occupation(self, UP):
        '''Parameters for initial occupation'''
        self.e_fermi = UP['e_fermi']*CoFa.eV_to_au           # Fermi energy
        self.temperature = UP['temperature']*CoFa.eV_to_au   # Temperature

    def __time_scales(self, UP):
        '''Time Scales'''
        self.T1 = UP['T1']*CoFa.fs_to_au                     # Density damping time
        self.T2 = UP['T2']*CoFa.fs_to_au                     # Coherence damping time
        self.t0 = UP['t0']*CoFa.fs_to_au
        self.dt = UP['dt']*CoFa.fs_to_au

    def __field(self, UP):
        '''Electrical Driving Field'''
        self.f = UP['f']*CoFa.THz_to_au                      # Driving pulse frequency

        if 'electric_field_function' in UP:
            self.electric_field_function = UP['electric_field_function']
            # Make first private to set it after params check in derived params
            self.__user_defined_field = True              # Disables all CUED specific field printouts

        else:
            self.electric_field_function = None           # Gets set in TimeContainers
            self.E0 = UP['E0']*CoFa.MVpcm_to_au              # Driving pulse field amplitude
            self.chirp = UP['chirp']*CoFa.THz_to_au          # Pulse chirp frequency
            self.sigma = UP['sigma']*CoFa.fs_to_au           # Gaussian pulse width
            self.phase = UP['phase']                         # Carrier-envelope phase
            self.__user_defined_field = False

    def __brillouin_zone(self, UP):
        '''Brillouin zone/Lattice'''
        self.BZ_type = UP['BZ_type']                         # Type of Brillouin zone
        self.Nk1 = UP['Nk1']                                 # kpoints in b1 direction
        self.Nk2 = UP['Nk2']                                 # kpoints in b2 direction

        if self.BZ_type == 'hexagon':
            self.align = UP['align']                         # E-field alignment
            self.angle_inc_E_field = None
            self.a = UP['a']                                 # Lattice spacing
        elif self.BZ_type == 'rectangle':
            self.align = None
            self.angle_inc_E_field = UP['angle_inc_E_field']
            self.length_BZ_ortho = UP['length_BZ_ortho']     # Size of the Brillouin zone in atomic units
            self.length_BZ_E_dir = UP['length_BZ_E_dir']     # -"-
        else:
            sys.exit("BZ_type needs to be either hexagon or rectangle.")

    def __optional(self, UP):
        '''Optional parameters or default values'''
        self.user_out = True                              # Command line progress output
        if 'user_out' in UP:
            self.user_out = UP['user_out']

        self.save_full = False                            # Save full density matrix
        if 'save_full' in UP:
            self.save_full = UP['save_full']

        self.save_latex_pdf = False                       # Save data as human readable PDF file
        if 'save_latex_pdf' in UP:
            self.save_latex_pdf = UP['save_latex_pdf']

        self.save_dm_t = False                            # Save density matrix at given points in time
        if 'save_dm_t' in UP:
            self.save_dm_t = UP['save_dm_t']

        self.save_fields = False
        if 'save_fields' in UP:
            self.save_fields = UP['save_fields']

        self.split_current = False                        # Save j^intra, j^anom, dP^inter/dt
        if 'split_current' in UP:
            self.split_current = UP['split_current']

        self.save_screening = False                       # Write screening files even with save_latex_pdf=False
        if 'save_screening' in UP:
            self.save_screening = UP['save_screening']

        self.plot_format = 'pdf'
        if 'plot_format' in UP:
            self.plot_format = UP['plot_format']

        self.gauge = 'length'                             # Gauge of the SBE Dynamics
        if 'gauge' in UP:
            self.gauge = UP['gauge']

        self.solver = 'nband'                             # 2 or n band solver
        if 'solver' in UP:
            self.solver = UP['solver']

        self.epsilon = 2e-5                               # Step size for num. derivatives
        if 'epsilon' in UP:
            self.epsilon = UP['epsilon']

        self.gidx = 1                                     # Index of real wave function entry (fixes gauge)
        if 'gidx' in UP:
            self.gidx = UP['gidx']

        self.save_anom = False                            # Calculate anomalous current
        if 'save_anom' in UP:
            self.save_anom = UP['save_anom']

        self.dm_dynamics_method = 'sbe'
        if 'dm_dynamics_method' in UP:
            self.dm_dynamics_method = UP['dm_dynamics_method']


        if self.dm_dynamics_method in ('sbe', 'semiclassics'):
            self.solver_method = 'bdf'                        # 'adams' non-stiff, 'bdf' stiff, 'rk4' Runge-Kutta 4th order
            if 'solver_method' in UP:
                self.solver_method = UP['solver_method']

        self.dk_order = 8                                 # Accuracy order of density-matrix k-deriv.
        if 'dk_order' in UP:
            self.dk_order = UP['dk_order']                   # with length gauge (avail: 2,4,6,8)
            if self.dk_order not in [2, 4, 6, 8]:
                sys.exit("dk_order needs to be either 2, 4, 6, or 8.")

        if self.dm_dynamics_method in ('series_expansion', 'EEA'):

            self.high_damping = False
            if 'high_damping' in UP:
                self.high_damping = UP['high_damping']

            self.first_order = True
            if 'first_order' in UP:
                self.first_order = UP['first_order']

            self.second_order = False 
            if 'second_order' in UP:
                self.second_order = UP['second_order']

            self.linear_response = False
            if 'linear_response' in UP:
                self.linear_response = UP['linear_response']

        self.precision = 'double'                         # Quadruple for reducing numerical noise
        if 'precision' in UP:
            self.precision = UP['precision']

        self.symmetric_insulator = False                  # Special flag for accurate insulator calc.
        if 'symmetric_insulator' in UP:
            self.symmetric_insulator = UP['symmetric_insulator']

        self.factor_freq_resolution = 1
        if 'factor_freq_resolution' in UP:
            self.factor_freq_resolution = UP['factor_freq_resolution']

        self.num_dimensions = 'automatic'                 # dimensionality for determining the prefactor (2*pi)^d of current
        if 'num_dimensions' in UP:
            self.num_dimensions = UP['num_dimensions']

        self.gabor_transformation = False
        if 'gabor_transformation' in UP:
            self.gabor_transformation = UP['gabor_transformation']

        self.fourier_window_function = 'hann'             # gaussian, parzen or hann
        if 'fourier_window_function' in UP:
            self.fourier_window_function = UP['fourier_window_function']

        if self.fourier_window_function == 'gaussian':
            if 'gaussian_window_width' in UP:
                self.gaussian_window_width = UP['gaussian_window_width']*CoFa.fs_to_au
            else:
                if self.__user_defined_field:
                    sys.exit("Gaussian needs a width (gaussian_window_width).")
                else:
                    self.gaussian_window_width = self.sigma
            if 'gaussian_center' in UP:
                self.gaussian_center = UP['gaussian_center']*CoFa.fs_to_au
            else:
                self.gaussian_center       = 0


        self.sheet_current = False
        if 'sheet_current' in UP:
            self.sheet_current = UP['sheet_current']

        self.degenerate_evals = False
        if 'degenerate_evals' in UP:
            self.degenerate_evals = UP['degenerate_evals']

        # Flag for Zeeman term
        self.Zeeman = False
        if 'Zeeman' in UP:
            self.Zeeman = UP['Zeeman']

        # Value of Magnetic field strength in Tesla if Zeeman flag is set true
        self.zeeman_strength = 0.0
        if 'zeeman_strength' in UP:
            self.zeeman_strength = UP['zeeman_strength']*0.5*CoFa.T_to_au

        #Flag for path parallelization, default value determined in main.py
        if 'path_parallelization' in UP:
            self.path_parallelization = UP['path_parallelization']

        #Add user defined header
        self.user_defined_header = ''
        if 'user_defined_header' in UP:
            self.user_defined_header = UP['user_defined_header']
            self.header = UP['user_defined_header'] + "_" + self.header

        self.do_fock = False
        if 'do_fock' in UP:
            self.do_fock = UP['do_fock']
        
        if self.do_fock = True  #split_paths = True and parallelize_over_points = False as default for Fock calculations
            if not 'split_paths' in UP:
                self.split_paths = True
            if not 'parallelize_over_points' in UP:
                self.parallelize_over_points = False


    def __check_user_params_for_wrong_arguments(self, UP):
        """
        Compare default paramaters with user parameters.
        If there are user parameters not defined in the parameters
        give a warning and halt the code.
        """
        default_params = self.__dict__.keys()
        user_params = UP.keys()
        diff_params = (default_params | user_params) - default_params

        if diff_params:
            print("Error: The following parameters have no effect inside the current run:")
            for param in diff_params:
                print(param, end=' ')
            print("Please delete them from params.py and rerun CUED. CUED will stop now.")
            print()
            sys.exit()

    def __check_for_conflicting_parameter_cominations(self, UP):
        """
        Give a warning and halt code if parameters are chosen such,
        that the code would not work or would use features that are
        not implemented yet
        """

        if self.split_paths == True:
            if self.parallelize_over_points = True:
                sys.exit('Paths can not be split if point parallelization is enabled')

        if self.do_fock == True:
            if self.solver_method != 'rk4':
                sys.exit('Fock calculations only run with unge-Kutta 4 ODE solver.')
            if self.solver != '2band':
                sys.exit('Fock calculations are only implemented for 2 band solver so far.')

    def __append_derived_parameters(self, UP):
        ##################################################
        ## The following parameters are derived parameters
        ## and can not be set in the params.py file
        ##################################################

        self.user_defined_field = self.__user_defined_field

        # Derived precision parameters
        if self.precision == 'double':
            self.type_real_np = np.float64
            self.type_complex_np = np.complex128
        elif self.precision == 'quadruple':
            self.type_real_np = np.float128
            self.type_complex_np = np.complex256
            if self.solver_method != 'rk4':
                sys.exit("Error: Quadruple precision only works with Runge-Kutta 4 ODE solver.")
        else:
            sys.exit("Only default or quadruple precision available.")

        # Derived initial condition
        self.e_fermi_eV = UP['e_fermi']
        self.temperature_eV = UP['temperature']

        # Derived driving field parameters
        self.f_THz = UP['f']
        if not self.user_defined_field:
            self.E0_MVpcm = UP['E0']
            self.chirp_THz = UP['chirp']
            self.sigma_fs = UP['sigma']

        # Derived time scale parameters
        self.gamma1 = 1/self.T1
        self.T1_fs = UP['T1']
        self.gamma1_dfs = 1/self.T1_fs

        self.gamma2 = 1/self.T2
        self.T2_fs = UP['T2']
        self.gamma2_dfs = 1/self.T2_fs

        self.t0_fs = UP['t0']
        self.tf = -self.t0
        self.tf_fs = -self.t0_fs

        self.dt = self.type_real_np(self.dt)
        self.dt_fs = self.type_real_np(UP['dt'])

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

        self.Nt_pdf_densmat = np.size(self.t_pdf_densmat)
