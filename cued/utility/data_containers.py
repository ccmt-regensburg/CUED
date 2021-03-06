import numpy as np
from cued.fields import make_electric_field

import cued.dipole
from cued.utility import MpiHelpers
from cued.utility import evaluate_njit_matrix
from cued.kpoint_mesh import hex_mesh, rect_mesh

class TimeContainers():
    def __init__(self, P):
        self.t = np.zeros(P.Nt, dtype=P.type_real_np)
        self.solution = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_real_np)
        self.solution_y_vec = np.zeros((((P.n)**2)*(P.Nk1)+1), dtype=P.type_complex_np)

        if P.save_full:
            self.solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, P.n, P.n), dtype=P.type_complex_np)

        self.A_field = np.zeros(P.Nt, dtype=P.type_real_np)
        self.E_field = np.zeros(P.Nt, dtype=P.type_real_np)

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
        if P.electric_field_function is None:
            self.electric_field = make_electric_field(P)
        else:
            self.electric_field = P.electric_field_function


class FrequencyContainers():
    pass


class SystemContainers():
    def __init__(self, P, sys):
        self.sys = sys

        self.Mpi = MpiHelpers()
        self.local_Nk2_idx_list = self.Mpi.get_local_idx(P.Nk2)

        # Form Brillouin Zone
        if P.BZ_type == 'hexagon':
            if P.align == 'K':
                self.E_dir = np.array([1, 0])
            elif P.align == 'M':
                self.E_dir = np.array([np.cos(np.radians(-30)),
                                       np.sin(np.radians(-30))])
            self.dk, self.kweight, self.paths = hex_mesh(P)

        elif P.BZ_type == 'rectangle':
            self.E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
                                   np.sin(np.radians(P.angle_inc_E_field))])
            self.dk, self.kweight, self.paths = rect_mesh(P, self)

        self.E_ort = np.array([self.E_dir[1], -self.E_dir[0]])

        # Calculate Eigensystem and Dipoles
        if P.hamiltonian_evaluation == 'ana':
            h_sym, ef_sym, wf_sym, _ediff_sym = sys.eigensystem(gidx=P.gidx)
            self.dipole = cued.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            self.curvature = cued.dipole.SymbolicCurvature(h_sym, self.dipole.Ax, self.dipole.Ay)
            P.n = 2

        if P.hamiltonian_evaluation == 'num':
            self.hnp = sys.hfjit
            P.n = np.size(evaluate_njit_matrix(self.hnp, kx=0, ky=0)[0, :, :], axis=0)

        if P.hamiltonian_evaluation == 'bandstructure':
            P.n = sys.n

        # Make in path containers
        self.dipole_path_x = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
        self.dipole_path_y = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)

        self.dipole_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
        self.dipole_ortho = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
        self.e_in_path = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)
        self.wf_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
