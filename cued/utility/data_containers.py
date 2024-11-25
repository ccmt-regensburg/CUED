import numpy as np
from cued.fields import make_electric_field_in_path, make_electric_field_ortho

import cued.dipole
from cued.utility import MpiHelpers
from cued.utility import evaluate_njit_matrix

class TimeContainers():
	def __init__(self, P):
		self.t = np.zeros(P.Nt, dtype=P.type_real_np)
		self.solution = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_complex_np)
		self.solution_y_vec = np.zeros((((P.n)**2)*(P.Nk1)+2), dtype=P.type_complex_np)

		if P.save_full:
			# Container for the full k-grid densities
			self.solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, P.n, P.n), dtype=P.type_complex_np)
			# Like the full solution this saves the full k-grid currents
			self.j_k_E_dir = np.empty((P.Nk1, P.Nk2, P.Nt), dtype=P.type_real_np)
			self.j_k_ortho = np.empty((P.Nk1, P.Nk2, P.Nt), dtype=P.type_real_np)
			# Containers used for k-dependent current buffers for one time step and path
			self.j_k_E_dir_path = np.empty(P.Nk1, dtype=P.type_real_np)
			self.j_k_ortho_path = np.empty(P.Nk1, dtype=P.type_real_np)

		self.A_field_in_path = np.zeros(P.Nt, dtype=P.type_real_np)
		self.E_field_in_path = np.zeros(P.Nt, dtype=P.type_real_np)
		self.A_field_ortho = np.zeros(P.Nt, dtype=P.type_real_np)
		self.E_field_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

		if P.sheet_current:
			self.j_E_dir = np.zeros([P.Nt, P.n_sheets, P.n_sheets], dtype=P.type_real_np)
			self.j_ortho = np.zeros([P.Nt, P.n_sheets, P.n_sheets], dtype=P.type_real_np)
			self.j_E_dir_full = np.zeros(P.Nt, dtype=P.type_real_np)
			self.j_ortho_full = np.zeros(P.Nt, dtype=P.type_real_np)
		else:
			self.j_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
			self.j_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

		if P.split_current:
			self.j_intra_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
			self.j_intra_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

			self.P_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
			self.P_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

			self.dtP_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
			self.dtP_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

			self.j_anom_ortho = np.zeros([P.Nt, P.n], dtype=P.type_real_np)
			self.j_anom_ortho_full = np.zeros(P.Nt, dtype=P.type_real_np)

		# Initialize electric_field, create rhs of ode and initialize solver
		if P.user_defined_field:
			self.electric_field_in_path = P.electric_field_function_in_path
			self.electric_field_ortho = P.electric_field_function_ortho
		else:
			self.electric_field_in_path = make_electric_field_in_path(P)
			self.electric_field_ortho = make_electric_field_ortho(P)

		if P.save_latex_pdf or P.save_dm_t:
			self.pdf_densmat = np.zeros((P.Nk1, P.Nk2, P.Nt_pdf_densmat, P.n, P.n), dtype=P.type_complex_np)
			self.t_pdf_densmat = np.zeros(P.Nt_pdf_densmat)

class FrequencyContainers():
	pass

class ScreeningContainers():
	def __init__(self, ff0, params_dims, plot_format):
		"""
		Container to generate plots of multiple parameter combinations e.g. CEP plots
		Parameters
		----------
		ff0 : np.ndarray
		    Frequencies of the system, normalized over the carrier frequency f0
		params_dims : tuple
		    Dimensions of screening parameters e.g. 2 E-fields, 2 chirps -> (2, 2)
		plot_format : string
		    Should the plot be saved as pdf or png?
		"""
		# frequencies
		self.ff0 = ff0
		self.plot_format = plot_format

		# All intensity data and current output data
		self.full_screening_data = np.empty(params_dims + (self.ff0.size, ), dtype=np.float64)
		self.screening_output = None

		# Parameter to be screened
		self._screening_parameter_name = None
		self.screening_parameter_name_plot_label = None
		self.screening_parameter_values = None

		# Filenames and LaTeX/Plotting/Saving related params
		self._screening_filename = None
		self.screening_filename_plot = None
		self.I_max_in_plotting_range = None

	@property
	def screening_parameter_name(self):
		return self._screening_parameter_name

	@screening_parameter_name.setter
	def screening_parameter_name(self, name):
		self._screening_parameter_name = name

		# First set to normal name if one of the special cases
		# is fulfilled change name
		self.screening_parameter_name_plot_label = name
		# System parameters
		if name == 'e_fermi':
			self.screening_parameter_name_plot_label = r'$\epsilon_{\scriptscriptstyle \mathrm{fermi}}$ in $\si{\electron\volt}$'
		if name == 'temperature':
			self.screening_parameter_name_plot_label = r'Temperature in $\si{\electron\volt}$'
		# Field variables
		if name == 'E0':
			self.screening_parameter_name_plot_label = r'$E_0$ in $\si{\mega\volt\per\cm}$'
		# if name == 'f':
		#     self.screening_parameter_name_plot_label = r'$f$ in $\si{\tera\hertz}$'
		if name == 'sigma':
			self.screening_parameter_name_plot_label = r'$\sigma$ in $\si{\femto\second}$'
		if name == 'chirp':
			self.screening_parameter_name_plot_label = r'$\omega_{\scriptscriptstyle \mathrm{chirp}}$ in $\si{\tera\hertz}$'
		if name == 'phase':
			self.screening_parameter_name_plot_label = r'phase $\phi$'
		# Time variables
		if name == 'T1':
			self.screening_parameter_name_plot_label = r'$T_1$ in $\si{\femto\second}$'
		if name == 'T2':
			self.screening_parameter_name_plot_label = r'$T_2$ in $\si{\femto\second}$'

	@property
	def screening_filename(self):
		return self._screening_filename

	@screening_filename.setter
	def screening_filename(self, filename):
		self._screening_filename = filename
		self.screening_filename_plot = filename + 'plot.' + self.plot_format
