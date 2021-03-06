from cued.dipole import diagonalize_path, dipole_elements_path
from cued.utility import evaluate_njit_matrix


def calculate_system_in_path(path, P, S):
    sys = S.sys

    # Retrieve the set of k-points for the current path
    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]

    if P.hamiltonian_evaluation == 'num':
        if P.do_semicl:
            0
        # Calculate the dot products E_dir.d_nm(k).
        # To be multiplied by E-field magnitude later.
        else:
            S.dipole_path_x, S.dipole_path_y = dipole_elements_path(path, P, S)
            S.e_in_path, S.wf_in_path = diagonalize_path(path, P, S)

    elif P.hamiltonian_evaluation == 'ana':
        if P.do_semicl:
            0
        else:
            # Calculate the dipole components along the path
            S.dipole_path_x = evaluate_njit_matrix(S.dipole.Axfjit, kx=kx_in_path, ky=ky_in_path)
            S.dipole_path_y = evaluate_njit_matrix(S.dipole.Ayfjit, kx=kx_in_path, ky=ky_in_path)
 
        S.e_in_path[:, 0] = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
        S.e_in_path[:, 1] = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)

        Ujit = sys.Ujit
        S.wf_in_path[:, 0, 0] = Ujit[0][0](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 0, 1] = Ujit[0][1](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 1, 0] = Ujit[1][0](kx=kx_in_path, ky=ky_in_path)
        S.wf_in_path[:, 1, 1] = Ujit[1][1](kx=kx_in_path, ky=ky_in_path)

    elif P.hamiltonian_evaluation == 'bandstructure':
        sys = S.sys

        S.dipole_path_x = evaluate_njit_matrix(sys.dipole_xjit, kx=kx_in_path,
                                        ky=ky_in_path, dtype=P.type_complex_np)
        S.dipole_path_y = evaluate_njit_matrix(sys.dipole_xjit, kx=kx_in_path,
                                        ky=ky_in_path, dtype=P.type_complex_np)

        for i in range(P.n):
            S.e_in_path[:, i] = sys.ejit[i](kx=kx_in_path, ky=ky_in_path)

    S.dipole_in_path = S.E_dir[0]*S.dipole_path_x + S.E_dir[1]*S.dipole_path_y
    S.dipole_ortho = S.E_ort[0]*S.dipole_path_x + S.E_ort[1]*S.dipole_path_y
