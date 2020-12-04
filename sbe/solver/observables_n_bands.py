import numpy as np
from sbe.utility import conversion_factors as co
from sbe.dipole import hamiltonian, diagonalize, derivative, dipole_elements

<<<<<<< HEAD
def current_in_path_hderiv(Nk_in_path, num_paths, Nt, density_matrix, n, paths, gidx, epsilon, path_idx):

    """
        Calculates the full and intraband current for a given path from eq. (67) in sbe_p01
=======
def hderiv(Nk_in_path, num_paths, n, paths, epsilon):
    """
        Function that calculates the derivative of the hamiltonian in kx- and ky-direction
>>>>>>> cb49baf721b8d8dcd68310b014d59562ba6a9bea

        Parameters
        ----------
        Nk_in_path : int
            number of k-points in x-direction (in the path)
        num_paths : int
            number of k-points in y-direction (number of paths)
<<<<<<< HEAD
        Nt : int
            number of time-steps
        density_matrix : np.ndarray
            solution of the semiconductor bloch equation (eq. 51 in sbe_p01)
        n : int
            number of bands
        paths : np.ndarray
            k-paths in the mesh
        gidx : int
            gauge index for the wf-gauge
        epsilon : float
            parameter for the derivative
        path_idx : int
            index of the current path

        Returns
        -------
        current_in_path : np.ndarray
            full current of the path with idx path_idx
        
    """

    # derivative dh/dk

=======
        n : int
            number of bands
        paths : np.ndarray
            k-paths along the derivative is evaluated
        epsilon : float
            parameter for the derivative
        
        Returns
        -------
        dhdkx : np.ndarray
            kx - derivative of the Hamiltonian
        dhdky : np.ndarray
            ky - derivative of the Hamiltonian
    """
>>>>>>> cb49baf721b8d8dcd68310b014d59562ba6a9bea
    epsilon = 0.15

    hgridplusx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    hgridminusx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    hgridplusy = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    hgridminusy = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

    dhdkx = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    dhdky = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

    for i in range(Nk_in_path):
        for j in range(num_paths):
            kx = paths[j, i, 0]
            ky = paths[j, i, 1]
            hgridplusx[i, j, :, :] = hamiltonian(kx + epsilon, ky)
            hgridminusx[i, j, :, :] = hamiltonian(kx - epsilon, ky)
            hgridplusy[i, j, :, :] = hamiltonian(kx, ky + epsilon)
            hgridminusy[i, j, :, :] = hamiltonian(kx, ky - epsilon)


    dhdkx = ( hgridplusx -  hgridminusx )/(2*epsilon)
    dhdky = ( hgridplusy -  hgridminusy )/(2*epsilon)

<<<<<<< HEAD
    # matrix elements <n k | dh/dk | n' k>
=======
    return dhdkx, dhdky

def matrix_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon):
    """
        Function that calculates the matrix elements < n, k| d_k h_in | n' k > 
        from eq. (67) in sbe_p01

        Parameters
        ----------
        Nk_in_path : int
            number of k-points in x-direction (in the path)
        num_paths : int
            number of k-points in y-direction (number of paths)
        n : int
            number of bands
        paths : np.ndarray
            k-paths along the matrix element is evaluated
        gidx : int
            gauge index for the wf-gauge
        epsilon : float
            parameter for the derivative
        
        Returns
        -------
        matrix_element_x, matrix_element_y : np.ndarray
            matrix elements form eq. 67 for each k-point and band combination
    """
    e, wf = diagonalize(Nk_in_path, num_paths, n, paths, gidx)
    dhdkx, dhdky = hderiv(Nk_in_path, num_paths, n, paths, epsilon)
>>>>>>> cb49baf721b8d8dcd68310b014d59562ba6a9bea

    matrix_element_x = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    matrix_element_y = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

    for i in range(Nk_in_path):
        for j in range(num_paths):
            buff = dhdkx[i,j,:,:] @ wf[i,j,:,:]
            matrix_element_x[i, j, :, :] = np.conjugate(wf[i, j, :, :].T) @ buff

            buff = dhdky[i,j,:,:] @ wf[i,j,:,:]
            matrix_element_y[i, j, :, :] = np.conjugate(wf[i, j, :, :].T) @ buff

<<<<<<< HEAD
    # full and intraband current via eq. (67) in sbe_p01

=======
def current_in_path_full(Nk_in_path, num_paths, Nt, density_matrix, n, paths, gidx, epsilon, path_idx):
    """
        Calculates the full current for a given path from eq. (67) in sbe_p01

        Parameters
        ----------
        Nk_in_path : int
            number of k-points in x-direction (in the path)
        num_paths : int
            number of k-points in y-direction (number of paths)
        Nt : int
            number of time-steps
        density_matrix : np.ndarray
            solution of the semiconductor bloch equation (eq. 51 in sbe_p01)
        n : int
            number of bands
        paths : np.ndarray
            k-paths in the mesh
        gidx : int
            gauge index for the wf-gauge
        epsilon : float
            parameter for the derivative
        path_idx : int
            index of the current path

        Returns
        -------
        current_in_path : np.ndarray
            full current of the path with idx path_idx
        
    """
>>>>>>> cb49baf721b8d8dcd68310b014d59562ba6a9bea
    current_in_path = np.zeros([Nt, 2], dtype=np.complex128)
    current_in_path_intraband = np.zeros([Nt, 2], dtype=np.complex128)
    melx = matrix_element_x[:, path_idx, :, :].reshape(Nk_in_path, n**2)
    mely = matrix_element_y[:, path_idx, :, :].reshape(Nk_in_path, n**2)

        # n = 2

    for i_t in range(Nt):
        #for j in range(n**2):
        current_in_path[i_t, 0] += - np.sum(melx[:, 0] * density_matrix[:, path_idx, i_t, 0].real)
        current_in_path[i_t, 1] += - np.sum(mely[:, 0] * density_matrix[:, path_idx, i_t, 0].real)        

        current_in_path[i_t, 0] += - np.sum(melx[:, 3] * density_matrix[:, path_idx, i_t, 3].real)           
        current_in_path[i_t, 1] += - np.sum(mely[:, 3] * density_matrix[:, path_idx, i_t, 3].real)

        current_in_path_intraband[i_t, :] += current_in_path[i_t, :]

    for i_t in range(Nt):
        current_in_path[i_t, 0] += -2*np.sum(np.real(melx[:, 1] * density_matrix[:, path_idx, i_t, 2])) 
        current_in_path[i_t, 1] += -2*np.sum(np.real(mely[:, 1] * density_matrix[:, path_idx, i_t, 2]))  

        # general n

    #for i_t in range(Nt):           #np.sum(?)
    #    for j in range(n**2):
    #        current_in_path[i_t, 0] += np.dot( melx[:, j], density_matrix[:, path_idx, i_t, j])
    #        current_in_path[i_t, 1] += np.dot( mely[:, j], density_matrix[:, path_idx, i_t, j])

    return current_in_path, current_in_path_intraband

<<<<<<< HEAD
def matrix_elements_dipoles(Nk_in_path, num_paths, n, paths, gidx, epsilon, dipole_in_path, e_in_path, path_idx):
    
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

    eplusx, wfplusx = diagonalize(Nk_in_path, num_paths, n, pathsplusx, gidx)
    eminusx, wfminusx = diagonalize(Nk_in_path, num_paths, n, pathsminusx, gidx)
    eplusy, wfplusy = diagonalize(Nk_in_path, num_paths, n, pathsplusy, gidx)
    eminusy, wfminusy = diagonalize(Nk_in_path, num_paths, n, pathsminusy, gidx)

    eplus2x, wfplus2x = diagonalize(Nk_in_path, num_paths, n, pathsplus2x, gidx)
    eminus2x, wfminus2x = diagonalize(Nk_in_path, num_paths, n, pathsminus2x, gidx)
    eplus2y, wfplus2y = diagonalize(Nk_in_path, num_paths, n, pathsplus2y, gidx)
    eminus2y, wfminus2y = diagonalize(Nk_in_path, num_paths, n, pathsminus2y, gidx)

    ederivx = ( - eplus2x + 8 * eplusx - 8 * eminusx + eminus2x)/(12*epsilon)
    ederivy = ( - eplus2y + 8 * eplusy - 8 * eminusy + eminus2y)/(12*epsilon)

    # matrix elements

    mel_x = np.zeros([Nk_in_path, n, n], dtype=np.complex128)
   # mel_y = np.zeros([Nk_in_path, n, n], dtype=np.float64)
    mel_x_intraband = np.zeros([Nk_in_path, n, n], dtype=np.float64)
   # mel_y_intraband = np.zeros([Nk_in_path, n, n], dtype=np.float64)

    for i in range(n):
        mel_x[:, i, i] += ederivx[:, path_idx, i]
        mel_x_intraband[:, i, i] += ederivx[:, path_idx, i]
       # matrix_element_y[:, i, i] += ederivx[:, path_idx, i]
       # matrix_element_y_intraband[:, i, i] += ederivx[:, path_idx, i]
        for j in range(n):
            mel_x[:, i, j] += 1j*dipole_in_path[:, i, j]*(e_in_path[:, j]-e_in_path[:, i])

    return mel_x.reshape(Nk_in_path, n, n), mel_x_intraband.reshape(Nk_in_path, n, n)

def current_in_path_dipole(Nk_in_path, num_paths, Nt, density_matrix, n, paths, gidx, epsilon, path_idx, dipole_in_path, e_in_path):

=======
def current_in_path_intraband(Nk_in_path, num_paths, Nt, density_matrix, n, paths, gidx, epsilon, path_idx):
    """
        Calculates the intraband current for a given path from eq. (67) in sbe_p01 (n = n')

        Parameters
        ----------
        Nk_in_path : int
            number of k-points in x-direction (in the path)
        num_paths : int
            number of k-points in y-direction (number of paths)
        Nt : int
            number of time-steps
        density_matrix : np.ndarray
            solution of the semiconductor bloch equation (eq. 51 in sbe_p01)
        n : int
            number of bands
        paths : np.ndarray
            k-paths in the mesh
        gidx : int
            gauge index for the wf-gauge
        epsilon : float
            parameter for the derivative
        path_idx : int
            index of the current path

        Returns
        -------
        current_in_path : np.ndarray
            intraband current of the path with idx path_idx
        
    """    
>>>>>>> cb49baf721b8d8dcd68310b014d59562ba6a9bea
    current_in_path = np.zeros([Nt, 2], dtype=np.complex128)
    current_in_path_intraband = np.zeros([Nt, 2], dtype=np.complex128)

    melx, mel_intraband= matrix_elements_dipoles(Nk_in_path, num_paths, n, paths, gidx, epsilon, dipole_in_path, e_in_path, path_idx)

    for i_t in range(Nt):
        for i in range(n):
            current_in_path[i_t, 0] += - np.sum(melx[:, i, i] * density_matrix[:, path_idx, i_t, i, i].real)
            current_in_path_intraband[i_t, 0] += - np.sum(mel_intraband[:, i, i] * density_matrix[:, path_idx, i_t, i, i].real)
            for j in range(n):        
                if i != j:
                    current_in_path[i_t, 0] += - np.sum(np.real(melx[:, i, j] * density_matrix[:, path_idx, i_t, j, i]))

        # current_in_path[i_t, 0] +=  -np.sum(melx[:, 0] * density_matrix[:, path_idx, i_t, 0].real)           
        # current_in_path[i_t, 0] +=  -2*np.sum(np.real(melx[:, 1] * density_matrix[:, path_idx, i_t, 2]))       
        # current_in_path[i_t, 0] +=  -np.sum(melx[:, 3] * density_matrix[:, path_idx, i_t, 3].real)

        # current_in_path_intraband[i_t, 0] +=  -np.sum(mel_intraband[:, 0] * density_matrix[:, path_idx, i_t, 0].real) 
        # current_in_path_intraband[i_t, 0] +=  -np.sum(mel_intraband[:, 3] * density_matrix[:, path_idx, i_t, 3].real) 

    return current_in_path, current_in_path_intraband