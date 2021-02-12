import numpy as np
from sbe.utility import ConversionFactors as co
from sbe.dipole import diagonalize, derivative
from sbe.utility import conditional_njit

def make_current_exact_path_hderiv(params, hamiltonian, paths, wf, E_dir, path_idx):
    """
        Function that calculates the exact current via eq. (79)
    """
    Nk_in_path = params.Nk1
    num_paths = params.Nk2
    n = params.n
    epsilon = 0.15
    type_complex_np = params.type_complex_np

    hgridplusx = np.empty([Nk_in_path, num_paths, n, n], dtype=type_complex_np)
    hgridminusx = np.empty([Nk_in_path, num_paths, n, n], dtype=type_complex_np)
    hgridplusy = np.empty([Nk_in_path, num_paths, n, n], dtype=type_complex_np)
    hgridminusy = np.empty([Nk_in_path, num_paths, n, n], dtype=type_complex_np)

    for i in range(Nk_in_path):
        for j in range(num_paths):
            kx = paths[j, i, 0]
            ky = paths[j, i, 1]
            hgridplusx[i, j, :, :] = hamiltonian(kx=kx+epsilon, ky=ky)
            hgridminusx[i, j, :, :] = hamiltonian(kx=kx-epsilon, ky=ky)
            hgridplusy[i, j, :, :] = hamiltonian(kx=kx, ky=ky+epsilon)
            hgridminusy[i, j, :, :] = hamiltonian(kx=kx, ky=ky-epsilon)
    dhdkx = ( hgridplusx -  hgridminusx )/(2*epsilon)
    dhdky = ( hgridplusy -  hgridminusy )/(2*epsilon)

    matrix_element_x = np.empty([Nk_in_path, n, n], dtype=type_complex_np)
    matrix_element_y = np.empty([Nk_in_path, n, n], dtype=type_complex_np)
    
    for i in range(Nk_in_path):
            buff = dhdkx[i, path_idx, :, :] @ wf[i, :, :]
            matrix_element_x[i, :, :] = np.conjugate(wf[i, :, :].T) @ buff

            buff = dhdky[i,path_idx,:,:] @ wf[i,:,:]
            matrix_element_y[i, :, :] = np.conjugate(wf[i, :, :].T) @ buff

    E_ort = np.array([E_dir[1], -E_dir[0]])

    mel_in_path = matrix_element_x[:, :, :] * E_dir[0] + matrix_element_y[:, :, :] * E_dir[1]
    mel_ortho = matrix_element_x[:, :, :] * E_ort[0] + matrix_element_y[:, :, :] * E_ort[1]
    
    @conditional_njit(type_complex_np)
    def current_exact_path_hderiv(solution):

        J_exact_E_dir = 0
        J_exact_ortho = 0

        for i_k in range(Nk_in_path):
            for i in range(n):
                J_exact_E_dir += - ( mel_in_path[i_k, i, i].real * solution[i_k, i, i].real )
                J_exact_ortho += - ( mel_ortho[i_k, i, i].real * solution[i_k, i, i].real )
                for j in range(n):
                    if i != j:
                        J_exact_E_dir += - np.real( mel_in_path[i_k, i, j] * solution[i_k, j, i] )
                        J_exact_ortho += - np.real( mel_ortho[i_k, i, j] * solution[i_k, j, i] )

        return J_exact_E_dir, J_exact_ortho
    return current_exact_path_hderiv



def make_polarization_inter_path(params, dipole_in_path, dipole_ortho):
    """
        Function that calculates the interband polarization from eq. (74)
    """
    n = params.n
    Nk1 = params.Nk1
    type_complex_np = params.type_complex_np

    @conditional_njit(type_complex_np)
    def polarization_inter_path(solution):

        P_inter_E_dir = 0
        P_inter_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                for j in range(n):
                    if i > j:
                        P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])    
                        P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])   
        return P_inter_E_dir, P_inter_ortho
    return polarization_inter_path


def make_intraband_current_path(params, hamiltonian, E_dir, paths, path_idx):
    """
        Function that calculates the intraband current from eq. (76 and 77) with or without the
        anomalous contribution via the Berry curvature
    """

    Nk1 = params.Nk1
    n = params.n
    gidx = params.gidx
    epsilon = params.epsilon
    type_complex_np = params.type_complex_np
    save_anom = params.save_anom

    # derivative of band structure

    pathsplusx = np.copy(paths)
    pathsplusx[:, :, 0] += epsilon
    pathsminusx = np.copy(paths)
    pathsminusx[:, :, 0] -= epsilon
    pathsplusy = np.copy(paths)
    pathsplusy[:, :, 1] += epsilon
    pathsminusy = np.copy(paths)
    pathsminusy[:, :, 1] -= epsilon

    pathsplus2x = np.copy(paths)
    pathsplus2x[:, :, 0] += 2*epsilon
    pathsminus2x = np.copy(paths)
    pathsminus2x[:, :, 0] -= 2*epsilon
    pathsplus2y = np.copy(paths)
    pathsplus2y[:, :, 1] += 2*epsilon
    pathsminus2y = np.copy(paths)
    pathsminus2y[:, :, 1] -= 2*epsilon

    eplusx, wfplusx = diagonalize(params, hamiltonian, pathsplusx)
    eminusx, wfminusx = diagonalize(params, hamiltonian, pathsminusx)
    eplusy, wfplusy = diagonalize(params, hamiltonian, pathsplusy)
    eminusy, wfminusy = diagonalize(params, hamiltonian, pathsminusy)

    eplus2x, wfplus2x = diagonalize(params, hamiltonian, pathsplus2x)
    eminus2x, wfminus2x = diagonalize(params, hamiltonian, pathsminus2x)
    eplus2y, wfplus2y = diagonalize(params, hamiltonian, pathsplus2y)
    eminus2y, wfminus2y = diagonalize(params, hamiltonian, pathsminus2y)

    ederivx = ( - eplus2x + 8 * eplusx - 8 * eminusx + eminus2x)/(12*epsilon)
    ederivy = ( - eplus2y + 8 * eplusy - 8 * eminusy + eminus2y)/(12*epsilon)

    # In path direction and orthogonal

    E_ort = np.array([E_dir[1], -E_dir[0]])
    ederiv_in_path = E_dir[0] * ederivx[:, path_idx, :] + E_dir[1] * ederivy[:, path_idx, :]
    ederiv_ortho = E_ort[0] * ederivx[:, path_idx, :] + E_ort[1] * ederivy[:, path_idx, :]

    @conditional_njit(type_complex_np)
    def current_intra_path(solution):

        J_intra_E_dir = 0
        J_intra_ortho = 0
        J_anom_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                J_intra_E_dir += - ederiv_in_path[k, i] * solution[k, i, i].real
                J_intra_ortho += - ederiv_ortho[k, i] * solution[k, i, i].real

                if save_anom:
                    J_anom_ortho += 0
                    print('J_anom not implemented')

        return J_intra_E_dir, J_intra_ortho, J_anom_ortho
    return current_intra_path
