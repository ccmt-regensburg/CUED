import numpy as np
import numpy.linalg as lin
import math

def diagonalize(params, n, hamiltonian, paths, gidx):
    """
        Diagonalize the n-dimensional Hamiltonian matrix on a 2-dimensional
        square k-grid with m*m k-points.
        The gauge fixes the entry of the wavefunction such, that the gidx-component is real

        Parameters
        ----------
        Nk_in_path : integer
            Number of k-points per path
        num_paths : integer
            Number of paths
        n : integer
            Number of bands
        paths : np.ndarray
            three dimensional array of all paths in
            the k-mesh (1st component: paths,
            2nd and 3rd component: x- and y- value of
            the k-points in the current path)
        gidx : integer
            Index of wavefunction component that will be gauged to be real

        Returns
        -------
        e : np.ndarray
            eigenenergies on each k-point
            first index: k-point in current path
            second index: index of current path
            third index: band index
        wf : np.ndarray
            wavefunctions on each k-point
            first index: k-point in current path
            second index: index of current path
            third index: component of wf
            fourth index: band index
    """
    Nk_in_path = params.Nk1
    num_paths = params.Nk2
    epsilon = params.epsilon
    e = np.empty([Nk_in_path, num_paths, n], dtype=np.float64)
    wf = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    for j in range(num_paths):
        for i in range(Nk_in_path):
            kx_in_path = paths[j, i, 0]
            ky_in_path = paths[j, i, 1]
            e[i, j], wf_buff = lin.eigh(hamiltonian(kx=kx_in_path, ky=ky_in_path))
            wf_gauged_entry = np.copy(wf_buff[gidx, :])
            wf_buff[gidx, :] = np.abs(wf_gauged_entry)
            wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
            wf[i, j] = wf_buff

    return e, wf


def derivative(params, n, hamiltonian, paths, gidx):

    Nk_in_path = params.Nk1
    num_paths = params.Nk2
    epsilon = params.epsilon

    xderivative = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    yderivative = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

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

    pathsplus3x = np.copy(paths)
    pathsplus3x[:, :, 0] += 3*epsilon
    pathsminus3x = np.copy(paths)
    pathsminus3x[:, :, 0] -= 3*epsilon
    pathsplus3y = np.copy(paths)
    pathsplus3y[:, :, 1] += 3*epsilon
    pathsminus3y = np.copy(paths)
    pathsminus3y[:, :, 1] -= 3*epsilon

    pathsplus4x = np.copy(paths)
    pathsplus4x[:, :, 0] += 4*epsilon
    pathsminus4x = np.copy(paths)
    pathsminus4x[:, :, 0] -= 4*epsilon
    pathsplus4y = np.copy(paths)
    pathsplus4y[:, :, 1] += 4*epsilon
    pathsminus4y = np.copy(paths)
    pathsminus4y[:, :, 1] -= 4*epsilon

    eplusx, wfplusx = diagonalize(params, n, hamiltonian, pathsplusx, gidx)
    eminusx, wfminusx = diagonalize(params, n, hamiltonian, pathsminusx, gidx)
    eplusy, wfplusy = diagonalize(params, n, hamiltonian, pathsplusy, gidx)
    eminusy, wfminusy = diagonalize(params, n, hamiltonian, pathsminusy, gidx)

    eplus2x, wfplus2x = diagonalize(params, n, hamiltonian, pathsplus2x, gidx)
    eminus2x, wfminus2x = diagonalize(params, n, hamiltonian, pathsminus2x, gidx)
    eplus2y, wfplus2y = diagonalize(params, n, hamiltonian, pathsplus2y, gidx)
    eminus2y, wfminus2y = diagonalize(params, n, hamiltonian, pathsminus2y, gidx)

    eplus3x, wfplus3x = diagonalize(params, n, hamiltonian, pathsplus3x, gidx)
    eminus3x, wfminus3x = diagonalize(params, n, hamiltonian, pathsminus3x, gidx)
    eplus3y, wfplus3y = diagonalize(params, n, hamiltonian, pathsplus3y, gidx)
    eminus3y, wfminus3y = diagonalize(params, n, hamiltonian, pathsminus3y, gidx)

    eplus4x, wfplus4x = diagonalize(params, n, hamiltonian, pathsplus4x, gidx)
    eminus4x, wfminus4x = diagonalize(params, n, hamiltonian, pathsminus4x, gidx)
    eplus4y, wfplus4y = diagonalize(params, n, hamiltonian, pathsplus4y, gidx)
    eminus4y, wfminus4y = diagonalize(params, n, hamiltonian, pathsminus4y, gidx)

    xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*(wfplusx - wfminusx) )/epsilon
    yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon

    return xderivative, yderivative


def dipole_elements(params, n, hamiltonian, paths, gidx):
    """
    Calculate the dipole elements

    Parameters
    ----------
    Nk_in_path : integer
        Number of k-points per path
    num_paths : integer
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

    Nk_in_path = params.Nk1
    num_paths = params.Nk2
    epsilon = params.epsilon

    e, wf = diagonalize(params, n, hamiltonian, paths, gidx)
    dwfkx, dwfky = derivative(params, n, hamiltonian, paths, gidx)

    dx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    dy = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

    for j in range(num_paths):
        for i in range(Nk_in_path):
            dx[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfkx[i, j, :, :])
            dy[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfky[i, j, :, :])

    return dx, dy
