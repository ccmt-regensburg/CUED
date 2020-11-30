import numpy as np
from sbe.utility import conversion_factors as co
from sbe.dipole import hamiltonian, diagonalize, derivative, dipole_elements

def hderiv(Nk_in_path, num_paths, n, paths, epsilon):
    """
        Function that calculates the derivative of the hamiltonian in kx- and ky-direction

        Parameters
        ----------
        Nk_in_path : int
            number of k-points in x-direction (in the path)
        num_paths : int
            number of k-points in y-direction (number of paths)
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

    matrix_element_x = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)
    matrix_element_y = np.empty([Nk_in_path, num_paths, n, n], dtype=np.complex128)

    for i in range(Nk_in_path):
        for j in range(num_paths):  #num. error irgendwo hier
            buff = dhdkx[i,j,:,:] @ wf[i,j,:,:]
            matrix_element_x[i, j, :, :] = np.conjugate(wf[i, j, :, :].T) @ buff

            buff = dhdky[i,j,:,:] @ wf[i,j,:,:]
            matrix_element_y[i, j, :, :] = np.conjugate(wf[i, j, :, :].T) @ buff
    
    return matrix_element_x, matrix_element_y

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
    current_in_path = np.zeros([Nt, 2], dtype=np.complex128)

    matrix_element_x, matrix_element_y = matrix_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon)

    melx = matrix_element_x[:, path_idx, :, :].reshape(Nk_in_path, n**2)
    mely = matrix_element_y[:, path_idx, :, :].reshape(Nk_in_path, n**2)

    for i_t in range(Nt):
        for i_k in range(Nk_in_path):     #length gauge only!
            #for j in range(n**2):
            current_in_path[i_t, 0] +=  melx[i_k, 0].real * density_matrix[i_k, path_idx, i_t, 0].real           #how about i_k -> : ?
            current_in_path[i_t, 1] +=  mely[i_k, 0].real * density_matrix[i_k, path_idx, i_t, 0].real
            current_in_path[i_t, 0] +=  2*np.real(melx[i_k, 1] * density_matrix[i_k, path_idx, i_t, 1])           #how about i_k -> : ?
            current_in_path[i_t, 1] +=  2*np.real(mely[i_k, 1] * density_matrix[i_k, path_idx, i_t, 1])           #how about i_k -> : ?                    current_in_path[i_t, 1] +=  mely[i_k, 1] * density_matrix[i_k, path_idx, i_t, 1]
            current_in_path[i_t, 0] +=  melx[i_k, 3].real * density_matrix[i_k, path_idx, i_t, 3].real           #how about i_k - : ?
            current_in_path[i_t, 1] +=  mely[i_k, 3].real * density_matrix[i_k, path_idx, i_t, 3].real

    #for i_t in range(Nt):           #np.sum(?)
    #    for j in range(n**2):
    #        current_in_path[i_t, 0] += np.dot( melx[:, j], density_matrix[:, path_idx, i_t, j])
    #        current_in_path[i_t, 1] += np.dot( mely[:, j], density_matrix[:, path_idx, i_t, j])

    return current_in_path

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
    current_in_path = np.zeros([Nt, 2], dtype=np.complex128)

    matrix_element_x, matrix_element_y = matrix_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon)

    for i_t in range(Nt):
        current_in_path[i_t, 0] += np.dot( matrix_element_x[:, path_idx, 0, 0].real, density_matrix[:, path_idx, i_t, 0 ].real )
        current_in_path[i_t, 0] += np.dot( matrix_element_x[:, path_idx, 1, 1].real, density_matrix[:, path_idx, i_t, 3 ].real )        
        current_in_path[i_t, 1] += np.dot( matrix_element_y[:, path_idx, 0, 0].real, density_matrix[:, path_idx, i_t, 0 ].real )
        current_in_path[i_t, 1] += np.dot( matrix_element_y[:, path_idx, 1, 1].real, density_matrix[:, path_idx, i_t, 3 ].real )

    return current_in_path        


def matrix_element_test(Nk_in_path, num_paths, n, paths, gidx, epsilon, dipole_x, e):
    
    matrix_element_x, matrix_element_y = matrix_elements(Nk_in_path, num_paths, n, paths, gidx, epsilon)
    for i in range(n):
        for j in range(n):
            test =  matrix_element_x[:, :, i, j] + 1j*dipole_x[:, :, i, j]*(e[:, :, j] - e[:, :, i]) 
            print(test, '\n\n')
    return 0