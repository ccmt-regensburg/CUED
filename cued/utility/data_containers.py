import numpy as np
from cued.fields import make_electric_field

import cued.dipole
from cued.utility import MpiHelpers
from cued.utility import evaluate_njit_matrix

class TimeContainers():
    def __init__(self, P):
        self.t = np.zeros(P.Nt, dtype=P.type_real_np)
        self.solution = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_complex_np)
        self.solution_y_vec = np.zeros((((P.n)**2)*(P.Nk1)+1), dtype=P.type_complex_np)

        if P.save_full:
            self.solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, P.n, P.n), dtype=P.type_complex_np)

        self.A_field = np.zeros(P.Nt, dtype=P.type_real_np)
        self.E_field = np.zeros(P.Nt, dtype=P.type_real_np)

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
            self.electric_field = P.electric_field_function
        else:
            self.electric_field = make_electric_field(P)

        if P.save_latex_pdf or P.save_dm_t:
            self.pdf_densmat = np.zeros((P.Nk1, P.Nk2, P.Nt_pdf_densmat, P.n, P.n), dtype=P.type_complex_np)
            self.t_pdf_densmat = np.zeros(P.Nt_pdf_densmat)

class FrequencyContainers():
    pass

class ScreeningContainers():
    def __init__(self, ff0, params_dims):
        # frequencies
        self.ff0 = ff0

        # All intensity data and current output data
        self.full_screening_data = np.empty(params_dims + (self.ff0.size, ), dtype=np.float64)
        self.screening_output = None

        # Parameter to be screened
        self.screening_parameter_name = None
        self.screening_parameter_values = None

        # Filenames and LaTeX/Plotting/Saving related params
        self._screening_filename = None
        self.screening_filename_plot = None
        self.I_max_in_plotting_range = None

    @property
    def screening_filename(self):
        return self._screening_filename

    @screening_filename.setter
    def screening_filename(self, filename):
        self._screening_filename = filename
        self.screening_filename_plot = filename + 'plot.pdf'