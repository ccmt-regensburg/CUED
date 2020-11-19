import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from params import params
from sbe.brillouin import hex_mesh, rect_mesh

# sigma_x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
# sigma_y = np.array([[0, -1j, 0],[1j, 0, 0], [0, 0, 0]])
# sigma_z = np.array([[1, 0, 0],[0, -1, 0], [0, 0, 0]])

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

# def theta(kx, ky):
#     theta = np.complex
#     if kx > 0:
#         theta = math.atan(ky/kx)
#     elif kx < 0:
#         if ky >= 0:
#             theta = math.atan(ky/kx) + np.pi
#         if ky < 0:
#             theta = math.atan(ky/kx) - np.pi
#     else:
#         if ky > 0:
#             theta = np.pi / 2
#         if ky < 0:
#             theta = - np.pi / 2
    
#     return theta

# def wavefunction(kx, ky, n):    
#     wf = np.empty([n, n], dtype=np.complex) #1st index: component, 2nd index: band
#     wf[0,0] = -1
#     wf[0,1] = 1
#     wf[1, :] = 1j*np.exp(1j*theta(kx, ky))
#     wf /= np.sqrt(2)

#     return wf

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
    A = 0.1974
    hmat = A*(ky*sigma_x - kx*sigma_y)

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
    wf = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
    
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

def derivative(Nk_in_path, num_paths, n, paths, gidx, epsilon):

    xderivative = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
    yderivative = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)

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

    xderivative = ( - wfplus2x + 8 * wfplusx - 8 * wfminusx + wfminus2x)/(12*epsilon)
    yderivative = ( - wfplus2y + 8 * wfplusy - 8 * wfminusy + wfminus2y)/(12*epsilon)

    return xderivative, yderivative

# def gradient(Nk_in_path, num_paths, n, paths, gidx, dk):
#     """
#         Calculate the kx- and ky- derivative of the wavefunction 
#         on the m x m square grid of k-points

#         Parameters
#         ----------
#         Nk_in_path : integer
#             Number of k-points per path
#         num_paths : integer
#             Number of paths
#         n : integer
#             Number of bands
#         wf : np.ndarray
#             wavefunctions on each k-point
#             first index: k-point in current path
#             second index: index of current path
#             third index: component of wf
#             fourth index: band index

#         Returns
#         -------
#         dwfkx, dwfky : np.ndarray
#             kx and ky derivative of the wavefunction on each k-point
#             first index: k-point in current path
#             second index: index of current path
#             third and fourth index: band indices
#     """
#     e, wf = diagonalize(Nk_in_path, num_paths, n, paths, gidx)
    
#     dwfkx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
#     dwfky = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
    
#     for i in range(n):
#         for j in range(n):
#             dwfky[:, :, i, j], dwfkx[:, :, i, j] = np.gradient(wf[:, :, i, j], dk)

#     return dwfkx, dwfky

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

    dx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
    dy = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex)
    
    for j in range(num_paths):
        for i in range(Nk_in_path):
            dx[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfkx[i, j, :, :])
            dy[i, j, :, :] = -1j*np.conjugate(wf[i, j, :, :]).T.dot(dwfky[i, j, :, :])
    
    return dx, dy
   
def plots(dx, dy, e, wf, num_paths, Nk_in_path, mesh): 
    """
    Plots the real part of the dipole-field for each combination of n and n'
    """

    plt.figure()

    plt.subplot(221)

    plt.quiver(mesh[:, 1], mesh[:, 0], np.real(dx[:, :, 0, 0]), np.real(dy[:, :, 0, 0]))

    plt.subplot(222)
    plt.quiver(mesh[:, 1], mesh[:, 0], np.real(dx[:, :, 1, 0]), np.real(dy[:, :, 1, 0]))

    plt.subplot(223)
    plt.quiver(mesh[:, 1], mesh[:, 0], np.real(dx[:, :, 0, 1]), np.real(dy[:, :, 0, 1]))

    plt.subplot(224)
    plt.quiver(mesh[:, 1], mesh[:, 0], np.real(dx[:, :, 1, 1]), np.real(dy[:, :, 1, 1]))

    plt.show()


     
if __name__ == "__main__":

    n = 2                                   #number of bands
    Nk_in_path = 2          #number of points per path
    num_paths = 2            #number of paths in k-space
    gidx = 0                                #index of gauged entry
    epsilon = 0.01

    angle_inc_E_field = params.angle_inc_E_field
    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                np.sin(np.radians(angle_inc_E_field))])

    dk, kweight, mesh, paths = rect_mesh(params, E_dir)
    #print(paths)
    e, wf = diagonalize(Nk_in_path, num_paths, n, paths, gidx)
    derx, dery = derivative(Nk_in_path, num_paths, n, paths, gidx, epsilon)
    #gradx, grady = gradient(Nk_in_path, num_paths, n, paths, gidx, dk)

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(wf[:, 0, 0, 1].real)
    # plt.plot(wf[:, 0, 1, 1].real)
    # plt.subplot(312)
    # plt.plot(derx[:, 0, 0 ,1].real)
    # plt.plot(derx[:, 0, 1, 1].real)
    # plt.subplot(313)
    # plt.plot(gradx[:, 0, 0 ,1].real)
    # plt.plot(gradx[:, 0, 1, 1].real)
    # plt.show()

    dx, dy = dipole_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon) 
    print('dx = ', dx, '\n dy = ', dy)
    #plots(dx, dy, e, wf, num_paths, Nk_in_path, mesh)
