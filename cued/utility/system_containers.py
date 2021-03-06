import numpy as np
import cued.dipole
from cued.kpoint_mesh import hex_mesh, rect_mesh
from cued.utility import evaluate_njit_matrix
from cued.utility import MpiHelpers

def system_properties(P, sys):
    class System():
        pass

    S = System()

    S.sys = sys
    
    S.Mpi = MpiHelpers()
    S.local_Nk2_idx_list = S.Mpi.get_local_idx(P.Nk2)

    # Form Brillouin Zone

    if P.BZ_type == 'hexagon':
        if P.align == 'K':
            S.E_dir = np.array([1, 0])
        elif P.align == 'M':
            S.E_dir = np.array([np.cos(np.radians(-30)),
                              np.sin(np.radians(-30))])
        S.dk, S.kweight, S.paths = hex_mesh(P)

    elif P.BZ_type == 'rectangle':
        S.E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
                          np.sin(np.radians(P.angle_inc_E_field))])
        S.dk, S.kweight, S.paths = rect_mesh(P, S)

    S.E_ort = np.array([S.E_dir[1], -S.E_dir[0]])

    # Calculate Eigensystem and Dipoles

    if P.hamiltonian_evaluation == 'ana':
        h_sym, ef_sym, wf_sym, _ediff_sym = sys.eigensystem(gidx=P.gidx)
        S.dipole = cued.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        S.curvature = cued.dipole.SymbolicCurvature(h_sym, S.dipole.Ax, S.dipole.Ay)
        P.n = 2

    if P.hamiltonian_evaluation == 'num':
        S.hnp = sys.hfjit
        P.n = np.size(evaluate_njit_matrix(S.hnp, kx=0, ky=0)[0, :, :], axis=0)

    if P.hamiltonian_evaluation == 'bandstructure':
        P.n = sys.n

    # Make in path containers
    
    S.dipole_path_x = np.zeros([P.Nk1, P.n, P.n], dtype = P.type_complex_np)
    S.dipole_path_y = np.zeros([P.Nk1, P.n, P.n], dtype = P.type_complex_np)

    S.dipole_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    S.dipole_ortho = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    S.e_in_path = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)
    S.wf_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)

    return S
