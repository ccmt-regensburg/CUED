import numpy as np
import numpy.linalg as lin
import math

from cued.utility import evaluate_njit_matrix

def diagonalize_path(path, P, S):

    n = P.n
    gidx = P.gidx
    Nk1 = P.Nk1
    Nk2 = P.Nk2
    epsilon = P.epsilon
    hamiltonian = S.hnp

    e_path = np.empty([Nk1, n], dtype=P.type_real_np)
    wf_path = np.empty([Nk1, n, n], dtype=P.type_complex_np)
    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]
    h_in_path = evaluate_njit_matrix(hamiltonian, kx_in_path, ky_in_path)
    for i in range(Nk1):
        e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
        wf_gauged_entry = np.copy(wf_buff[gidx, :])
        wf_buff[gidx, :] = np.abs(wf_gauged_entry)
        wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
        wf_path[i] = wf_buff

    return e_path, wf_path

def derivative_path(path, P, S):

    Nk1 = P.Nk1
    Nk2 = P.Nk2
    epsilon = P.epsilon
    n = P.n
    gidx = P.gidx

    xderivative = np.empty([Nk1, n, n], dtype=P.type_complex_np)
    yderivative = np.empty([Nk1, n, n], dtype=P.type_complex_np)

    pathplusx = np.copy(path)
    pathplusx[:, 0] += epsilon
    pathminusx = np.copy(path)
    pathminusx[:, 0] -= epsilon
    pathplusy = np.copy(path)
    pathplusy[:, 1] += epsilon
    pathminusy = np.copy(path)
    pathminusy[:, 1] -= epsilon

    pathplus2x = np.copy(path)
    pathplus2x[:, 0] += 2*epsilon
    pathminus2x = np.copy(path)
    pathminus2x[:, 0] -= 2*epsilon
    pathplus2y = np.copy(path)
    pathplus2y[:, 1] += 2*epsilon
    pathminus2y = np.copy(path)
    pathminus2y[:, 1] -= 2*epsilon

    pathplus3x = np.copy(path)
    pathplus3x[:, 0] += 3*epsilon
    pathminus3x = np.copy(path)
    pathminus3x[:, 0] -= 3*epsilon
    pathplus3y = np.copy(path)
    pathplus3y[:, 1] += 3*epsilon
    pathminus3y = np.copy(path)
    pathminus3y[:, 1] -= 3*epsilon

    pathplus4x = np.copy(path)
    pathplus4x[:, 0] += 4*epsilon
    pathminus4x = np.copy(path)
    pathminus4x[:, 0] -= 4*epsilon
    pathplus4y = np.copy(path)
    pathplus4y[:, 1] += 4*epsilon
    pathminus4y = np.copy(path)
    pathminus4y[:, 1] -= 4*epsilon

    eplusx, wfplusx = diagonalize_path(pathplusx, P, S)
    eminusx, wfminusx = diagonalize_path(pathminusx, P, S)
    eplusy, wfplusy = diagonalize_path(pathplusy, P, S)
    eminusy, wfminusy = diagonalize_path(pathminusy, P, S)

    eplus2x, wfplus2x = diagonalize_path(pathplus2x, P, S)
    eminus2x, wfminus2x = diagonalize_path(pathminus2x, P, S)
    eplus2y, wfplus2y = diagonalize_path(pathplus2y, P, S)
    eminus2y, wfminus2y = diagonalize_path(pathminus2y, P, S)

    eplus3x, wfplus3x = diagonalize_path(pathplus3x, P, S)
    eminus3x, wfminus3x = diagonalize_path(pathminus3x, P, S)
    eplus3y, wfplus3y = diagonalize_path(pathplus3y, P, S)
    eminus3y, wfminus3y = diagonalize_path(pathminus3y, P, S)

    eplus4x, wfplus4x = diagonalize_path(pathplus4x, P, S)
    eminus4x, wfminus4x = diagonalize_path(pathminus4x, P, S)
    eplus4y, wfplus4y = diagonalize_path(pathplus4y, P, S)
    eminus4y, wfminus4y = diagonalize_path(pathminus4y, P, S)

    xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*(wfplusx - wfminusx) )/epsilon
    yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon

    return xderivative, yderivative


def dipole_elements_path(path, P, S):
    """
    Calculate the dipole elements

    Parameters
    ----------
    Nk1 : integer
        Number of k-points per path
    Nk2 : integer
        Number of paths
    n : integer
        Number of bands
    kxvalues, kyvalues : np.ndarray
        array with kx- and ky- values of the k-grid
    wf : np.ndarray
        wavefunctions on each k-point
    dwfkx, dwfky : np.ndarray
        kx and ky derivative of the wavefunction on each k-point
        first index: k-point in current path
        second index: index of current path
        third and fourth index: band indices

    Returns
    -------
    dx, dy : np.ndarray
        x and y component of the Dipole-field d_nn'(k) (Eq. (37)) for each k-point
    """

    Nk1 = P.Nk1
    epsilon = P.epsilon
    gidx = P.gidx
    n = P.n

    e_path, wf_path = diagonalize_path(path, P, S)
    dwfkx_path, dwfky_path = derivative_path(path, P, S)

    dx_path = np.empty([Nk1, n, n], dtype=np.complex128)
    dy_path = np.empty([Nk1, n, n], dtype=np.complex128)

    for i in range(Nk1):
        dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
        dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

    return dx_path, dy_path

# def diagonalize(P, S):
#     """
#         Diagonalize the n-dimensional Hamiltonian matrix on a 2-dimensional
#         square k-grid with m*m k-points.
#         The gauge fixes the entry of the wavefunction such, that the gidx-component is real

#         Parameters
#         ----------
#         Nk1 : integer
#             Number of k-points per path
#         Nk2 : integer
#             Number of paths
#         n : integer
#             Number of bands
#         paths : np.ndarray
#             three dimensional array of all paths in
#             the k-mesh (1st component: paths,
#             2nd and 3rd component: x- and y- value of
#             the k-points in the current path)
#         gidx : integer
#             Index of wavefunction component that will be gauged to be real

#         Returns
#         -------
#         e : np.ndarray
#             eigenenergies on each k-point
#             first index: k-point in current path
#             second index: index of current path
#             third index: band index
#         wf : np.ndarray
#             wavefunctions on each k-point
#             first index: k-point in current path
#             second index: index of current path
#             third index: component of wf
#             fourth index: band index
#     """
#     n = P.n
#     gidx = P.gidx
#     Nk1 = P.Nk1
#     Nk2 = P.Nk2
#     epsilon = P.epsilon
#     hamiltonian = S.hnp
#     paths = S.paths

#     e = np.empty([Nk1, Nk2, n], dtype=np.float64)
#     wf = np.empty([Nk1, Nk2, n, n], dtype=np.complex128)
#     for j in range(Nk2):
#         kx_in_path = paths[j, :, 0]
#         ky_in_path = paths[j, :, 1]
#         h_in_path = evaluate_njit_matrix(hamiltonian, kx_in_path, ky_in_path)
#         for i in range(Nk1):
#             e[i, j], wf_buff = lin.eigh(h_in_path[i, :, :])
#             wf_gauged_entry = np.copy(wf_buff[gidx, :])
#             wf_buff[gidx, :] = np.abs(wf_gauged_entry)
#             wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
#             wf[i, j] = wf_buff

#     return e, wf


# def derivative(P, S):

#     Nk1 = P.Nk1
#     Nk2 = P.Nk2
#     epsilon = P.epsilon
#     n = P.n
#     gidx = P.gidx
#     paths = S.paths

#     xderivative = np.empty([Nk1, Nk2, n, n], dtype=np.complex128)
#     yderivative = np.empty([Nk1, Nk2, n, n], dtype=np.complex128)

#     pathsplusx = np.copy(paths)
#     pathsplusx[:, :, 0] += epsilon
#     pathsminusx = np.copy(paths)
#     pathsminusx[:, :, 0] -= epsilon
#     pathsplusy = np.copy(paths)
#     pathsplusy[:, :, 1] += epsilon
#     pathsminusy = np.copy(paths)
#     pathsminusy[:, :, 1] -= epsilon

#     pathsplus2x = np.copy(paths)
#     pathsplus2x[:, :, 0] += 2*epsilon
#     pathsminus2x = np.copy(paths)
#     pathsminus2x[:, :, 0] -= 2*epsilon
#     pathsplus2y = np.copy(paths)
#     pathsplus2y[:, :, 1] += 2*epsilon
#     pathsminus2y = np.copy(paths)
#     pathsminus2y[:, :, 1] -= 2*epsilon

#     pathsplus3x = np.copy(paths)
#     pathsplus3x[:, :, 0] += 3*epsilon
#     pathsminus3x = np.copy(paths)
#     pathsminus3x[:, :, 0] -= 3*epsilon
#     pathsplus3y = np.copy(paths)
#     pathsplus3y[:, :, 1] += 3*epsilon
#     pathsminus3y = np.copy(paths)
#     pathsminus3y[:, :, 1] -= 3*epsilon

#     pathsplus4x = np.copy(paths)
#     pathsplus4x[:, :, 0] += 4*epsilon
#     pathsminus4x = np.copy(paths)
#     pathsminus4x[:, :, 0] -= 4*epsilon
#     pathsplus4y = np.copy(paths)
#     pathsplus4y[:, :, 1] += 4*epsilon
#     pathsminus4y = np.copy(paths)
#     pathsminus4y[:, :, 1] -= 4*epsilon

#     S.paths = pathsplusx
#     eplusx, wfplusx = diagonalize(P, S)
#     S.paths = pathsminusx
#     eminusx, wfminusx = diagonalize(P, S)
#     S.paths = pathsplusy
#     eplusy, wfplusy = diagonalize(P, S)
#     S.paths = pathsminusy
#     eminusy, wfminusy = diagonalize(P, S)

#     S.paths = pathsplus2x
#     eplus2x, wfplus2x = diagonalize(P, S)
#     S.paths = pathsminus2x
#     eminus2x, wfminus2x = diagonalize(P, S)
#     S.paths = pathsplus2y
#     eplus2y, wfplus2y = diagonalize(P, S)
#     S.paths = pathsminus2y
#     eminus2y, wfminus2y = diagonalize(P, S)

#     S.paths = pathsplus3x
#     eplus3x, wfplus3x = diagonalize(P, S)
#     S.paths = pathsminus3x
#     eminus3x, wfminus3x = diagonalize(P, S)
#     S.paths = pathsplus3y
#     eplus3y, wfplus3y = diagonalize(P, S)
#     S.path3s = pathsminus3y
#     eminus3y, wfminus3y = diagonalize(P, S)

#     S.paths = pathsplus4x
#     eplus4x, wfplus4x = diagonalize(P, S)
#     S.paths = pathsminus4x
#     eminus4x, wfminus4x = diagonalize(P, S)
#     S.paths = pathsplus4y
#     eplus4y, wfplus4y = diagonalize(P, S)
#     S.paths = pathsminus2y
#     eminus4y, wfminus4y = diagonalize(P, S)

#     S.paths = paths # reset to original path

#     xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*(wfplusx - wfminusx) )/epsilon
#     yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon

#     return xderivative, yderivative


# def dipole_elements(P, S):
#     """
#     Calculate the dipole elements

#     Parameters
#     ----------
#     Nk1 : integer
#         Number of k-points per path
#     Nk2 : integer
#         Number of paths
#     n : integer
#         Number of bands
#     kxvalues, kyvalues : np.ndarray
#         array with kx- and ky- values of the k-grid
#     wf : np.ndarray
#         wavefunctions on each k-point
#     dwfkx, dwfky : np.ndarray
#         kx and ky derivative of the wavefunction on each k-point
#         first index: k-point in current path
#         second index: index of current path
#         third and fourth index: band indices

#     Returns
#     -------
#     dx, dy : np.ndarray
#         x and y component of the Dipole-field d_nn'(k) (Eq. (37)) for each k-point
#     """

#     Nk1 = P.Nk1
#     Nk2 = P.Nk2
#     epsilon = P.epsilon
#     gidx = P.gidx
#     n = P.n

#     e, wf = diagonalize(P, S)
#     dwfkx, dwfky = derivative(P, S)

#     dx = np.empty([Nk1, Nk2, n, n], dtype=np.complex128)
#     dy = np.empty([Nk1, Nk2, n, n], dtype=np.complex128)

#     for j in range(Nk2):
#         for i in range(Nk1):
#             dx[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfkx[i, j, :, :])
#             dy[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfky[i, j, :, :])

#     return dx, dy
