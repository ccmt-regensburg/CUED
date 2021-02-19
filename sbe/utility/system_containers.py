import numpy as np
import sbe.dipole
from sbe.dipole import diagonalize, dipole_elements
from sbe.kpoint_mesh import hex_mesh, rect_mesh


def system_properties(P, sys):
    class System():
        pass

    S = System()

    S.sys = sys

    # Form Brillouin Zone

    if P.BZ_type == 'hexagon':
        S.dk, S.kweight, S.paths = hex_mesh(P)
        if P.align == 'K':
            S.E_dir = np.array([1, 0])
        elif P.align == 'M':
            S.E_dir = np.array([np.cos(np.radians(-30)),
                              np.sin(np.radians(-30))])
    elif P.BZ_type == 'rectangle':
        S.E_dir = np.array([np.cos(np.radians(P.angle_inc_E_field)),
                          np.sin(np.radians(P.angle_inc_E_field))])
        S.dk, S.kweight, S.paths = rect_mesh(P, S)

    S.E_ort = np.array([S.E_dir[1], -S.E_dir[0]])

    # Calculate Eigensystem and Dipoles

    S.hnp = sys.numpy_hamiltonian()

    if P.system == 'ana':

        h_sym, ef_sym, wf_sym, _ediff_sym = sys.eigensystem(gidx=P.gidx)
        S.dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        S.curvature = sbe.dipole.SymbolicCurvature(h_sym, S.dipole.Ax, S.dipole.Ay)
        P.n = 2

    if P.system == 'num':
        P.n = np.size(S.hnp(kx=0, ky=0)[:, 0])
        S.dipole_x, S.dipole_y = dipole_elements(P, S)
        S.e, S.wf = diagonalize(P, S)
        S.curvature = 0   
        S.dipole = 0

    # Make in path containers

    S.dipole_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    
    S.dipole_ortho = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    
    S.e_in_path = np.zeros([P.Nk1, P.n], dtype=P.type_real_np)  

    S.wf_in_path = np.zeros([P.Nk1, P.n, P.n], dtype=P.type_complex_np)
    
    return S