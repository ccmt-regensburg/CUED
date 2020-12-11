import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from sbe.brillouin import hex_mesh, rect_mesh

#sigma_x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
#sigma_y = np.array([[0, -1j, 0],[1j, 0, 0], [0, 0, 0]])
#sigma_z = np.array([[1, 0, 0],[0, -1, 0], [0, 0, 0]])

sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

def theta(kx, ky):
    theta = np.complex128
    if kx > 0:
        theta = math.atan(ky/kx)
    elif kx < 0:
        if ky >= 0:
            theta = math.atan(ky/kx) + np.pi
        if ky < 0:
            theta = math.atan(ky/kx) - np.pi
    else:
        if ky > 0:
            theta = np.pi / 2
        if ky < 0:
            theta = - np.pi / 2
    
    return theta

def wavefunction(kx, ky, n):    
    wf = np.empty([n, n], dtype=np.complex128) #1st index: component, 2nd index: band
    wf[0,0] = -1
    wf[0,1] = 1
    wf[1, :] = 1j*np.exp(1j*theta(kx, ky))
    wf /= np.sqrt(2)

    return wf

def hamiltonian(kx, ky):
    """
        Build the 2-dimensional Dirac-hamiltonian matrix at the point (kx, ky)

        Parameters
        ----------
        kx, ky : 
            coordinates of the k-point on the rectangular mesh

        Returns
        -------
        hmat : np.ndarray
            Hamiltonian in matrix-form at point (kx, ky)
    """

    A = 0.19732
    m = 0
    hmat = A*(ky*sigma_x - kx*sigma_y) + m*sigma_z

    return hmat

def diagonalize(Nk_in_path, num_paths, n, paths, gidx):   #gidx = index of gauged entry
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
    e = np.empty([Nk_in_path, num_paths, n], dtype=np.float64)
    wf = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    
    #print(paths)

    for j in range(num_paths):
        for i in range(Nk_in_path):
            kx = paths[j, i, 0]
            ky = paths[j, i, 1]
            e[i, j], wf_buff = lin.eigh(hamiltonian(kx, ky))
            wf_gauged_entry = np.copy(wf_buff[gidx, :])
            wf_buff[gidx, :] = np.abs(wf_gauged_entry)
            wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
            wf[i, j] = wf_buff
            #wf[i, j] = wavefunction(kx, ky, n)

    return e, wf

def ederivative(Nk_in_path, num_paths, n, paths, gidx, epsilon):

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

    eplusx, wfplusx = diagonalize(Nk_in_path, num_paths, n, pathsplusx, gidx)
    eminusx, wfminusx = diagonalize(Nk_in_path, num_paths, n, pathsminusx, gidx)
    eplusy, wfplusy = diagonalize(Nk_in_path, num_paths, n, pathsplusy, gidx)
    eminusy, wfminusy = diagonalize(Nk_in_path, num_paths, n, pathsminusy, gidx)

    eplus2, wfplus2x = diagonalize(Nk_in_path, num_paths, n, pathsplus2x, gidx)
    eminus2x, wfminus2x = diagonalize(Nk_in_path, num_paths, n, pathsminus2x, gidx)
    eplus2y, wfplus2y = diagonalize(Nk_in_path, num_paths, n, pathsplus2y, gidx)
    eminus2y, wfminus2y = diagonalize(Nk_in_path, num_paths, n, pathsminus2y, gidx)

    exderivative = ( - eplus2x + 8 * eplusx - 8 * eminusx + eminus2x)/(12*epsilon)
    eyderivative = ( - eplus2y + 8 * eplusy - 8 * eminusy + eminus2y)/(12*epsilon)

    return exderivative, eyderivative

def derivative(Nk_in_path, num_paths, n, paths, gidx, epsilon):

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

    eplusx, wfplusx = diagonalize(Nk_in_path, num_paths, n, pathsplusx, gidx)
    eminusx, wfminusx = diagonalize(Nk_in_path, num_paths, n, pathsminusx, gidx)
    eplusy, wfplusy = diagonalize(Nk_in_path, num_paths, n, pathsplusy, gidx)
    eminusy, wfminusy = diagonalize(Nk_in_path, num_paths, n, pathsminusy, gidx)

    eplus2x, wfplus2x = diagonalize(Nk_in_path, num_paths, n, pathsplus2x, gidx)
    eminus2x, wfminus2x = diagonalize(Nk_in_path, num_paths, n, pathsminus2x, gidx)
    eplus2y, wfplus2y = diagonalize(Nk_in_path, num_paths, n, pathsplus2y, gidx)
    eminus2y, wfminus2y = diagonalize(Nk_in_path, num_paths, n, pathsminus2y, gidx)

    eplus3x, wfplus3x = diagonalize(Nk_in_path, num_paths, n, pathsplus3x, gidx)
    eminus3x, wfminus3x = diagonalize(Nk_in_path, num_paths, n, pathsminus3x, gidx)
    eplus3y, wfplus3y = diagonalize(Nk_in_path, num_paths, n, pathsplus3y, gidx)
    eminus3y, wfminus3y = diagonalize(Nk_in_path, num_paths, n, pathsminus3y, gidx)

    eplus4x, wfplus4x = diagonalize(Nk_in_path, num_paths, n, pathsplus4x, gidx)
    eminus4x, wfminus4x = diagonalize(Nk_in_path, num_paths, n, pathsminus4x, gidx)
    eplus4y, wfplus4y = diagonalize(Nk_in_path, num_paths, n, pathsplus4y, gidx)
    eminus4y, wfminus4y = diagonalize(Nk_in_path, num_paths, n, pathsminus4y, gidx)

    xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*(wfplusx - wfminusx) )/epsilon
    yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon

    return xderivative, yderivative

def dipole_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon):    
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
    e, wf = diagonalize(Nk_in_path, num_paths, n, paths, gidx)
    dwfkx, dwfky = derivative(Nk_in_path, num_paths, n, paths, gidx, epsilon)
    #dwfkx, dwfky = gradient(Nk_in_path, num_paths, n, paths, gidx, dk)

    dx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    dy = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    
    for j in range(num_paths):
        for i in range(Nk_in_path):
            dx[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfkx[i, j, :, :])
            dy[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfky[i, j, :, :])

    return dx, dy
