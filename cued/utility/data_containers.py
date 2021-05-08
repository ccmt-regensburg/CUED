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

            self.j_anom_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

        # Initialize electric_field, create rhs of ode and initialize solver
        if P.user_defined_field:
            self.electric_field = P.electric_field_function
        else:
            self.electric_field = make_electric_field(P)

        if P.save_latex_pdf:
            self.pdf_densmat = np.zeros((P.Nk1, P.Nk2, P.Nt_pdf_densmat, P.n, P.n), dtype=P.type_complex_np)
            self.t_pdf_densmat = np.zeros(P.Nt_pdf_densmat)

class FrequencyContainers():
    pass
