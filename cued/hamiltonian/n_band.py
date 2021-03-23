import numpy as np
import sympy as sp
import numpy.linalg as lin
import math

from cued.utility import evaluate_njit_matrix, list_to_njit_functions, matrix_to_njit_functions


class NBandHamiltonianSystem():

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    def __init__(self, h):
        
        self.system = 'num'

        self.h = h
        self.hsymbols = self.h.free_symbols
        self.hderiv = self.__hamiltonian_derivatives()

        self.hfjit = matrix_to_njit_functions(self.h, self.hsymbols)
        self.hderivfjit = [matrix_to_njit_functions(hd, self.hsymbols)
                           for hd in self.hderiv]
        
        self.n = np.size(evaluate_njit_matrix(self.hfjit, kx=0, ky=0)[0, :, :], axis=0)

        self.U = None             # Normalised eigenstates

        self.e_in_path = None   #set when eigensystem_dipole_path is called
        self.wf_in_path = None

        self.dipole_path_x = None   # set when eigensystem_dipole_path is called
        self.dipole_path_y = None
                           
        self.dipole_in_path = None
        self.dipole_ortho = None
        
    def __hamiltonian_derivatives(self):
        return [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]


    def __derivative_path(self, path, P):
        
        epsilon = P.epsilon
        pathlen = path[:, 0].size

        xderivative = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)
        yderivative = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)

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

        eplusx, wfplusx = self.diagonalize_path(pathplusx, P)
        eminusx, wfminusx = self.diagonalize_path(pathminusx, P)
        eplusy, wfplusy = self.diagonalize_path(pathplusy, P)
        eminusy, wfminusy = self.diagonalize_path(pathminusy, P)

        eplus2x, wfplus2x = self.diagonalize_path(pathplus2x, P)
        eminus2x, wfminus2x = self.diagonalize_path(pathminus2x, P)
        eplus2y, wfplus2y = self.diagonalize_path(pathplus2y, P)
        eminus2y, wfminus2y = self.diagonalize_path(pathminus2y, P)

        eplus3x, wfplus3x = self.diagonalize_path(pathplus3x, P)
        eminus3x, wfminus3x = self.diagonalize_path(pathminus3x, P)
        eplus3y, wfplus3y = self.diagonalize_path(pathplus3y, P)
        eminus3y, wfminus3y = self.diagonalize_path(pathminus3y, P)

        eplus4x, wfplus4x = self.diagonalize_path(pathplus4x, P)
        eminus4x, wfminus4x = self.diagonalize_path(pathminus4x, P)
        eplus4y, wfplus4y = self.diagonalize_path(pathplus4y, P)
        eminus4y, wfminus4y = self.diagonalize_path(pathminus4y, P)

        xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*(wfplusx - wfminusx) )/epsilon
        yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon

        return xderivative, yderivative

    def diagonalize_path(self, path, P):

        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        pathlen = path[:, 0].size
        e_path = np.empty([pathlen, P.n], dtype=P.type_real_np)
        wf_path = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)
        h_in_path = evaluate_njit_matrix(self.hfjit, kx_in_path, ky_in_path)
  
        for i in range(pathlen):
            e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
            wf_gauged_entry = np.copy(wf_buff[P.gidx, :])
            wf_buff[P.gidx, :] = np.abs(wf_gauged_entry)
            wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == P.gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
            wf_path[i] = wf_buff

        return e_path, wf_path


    def eigensystem_dipole_path(self, path, P):
        '''
            Dipole Elements of the Hamiltonian for a given path
        '''

        pathlen = path[:, 0].size
        e_path, wf_path = self.diagonalize_path(path, P)
        dwfkx_path, dwfky_path = self.__derivative_path(path, P)

        dx_path = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)
        dy_path = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)

        if not P.do_semicl:
            for i in range(pathlen):
                dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
                dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

        self.e_in_path = e_path
        self.wf_in_path = wf_path
        self.dipole_path_x = dx_path
        self.dipole_path_y = dy_path
        self.dipole_in_path = P.E_dir[0]*dx_path + P.E_dir[1]*dy_path
        self.dipole_ortho = P.E_ort[0]*dx_path + P.E_ort[1]*dy_path