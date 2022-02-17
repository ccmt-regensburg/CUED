import numpy as np
import sympy as sp
import numpy.linalg as lin
import math
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger
from cued.utility import evaluate_njit_matrix, list_to_njit_functions, matrix_to_njit_functions

class NBandHamiltonianSystem():

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    def __init__(self, h, n_sheets=1, degenerate_eigenvalues=False, ana_e=None, ana_wf=None):

        self.system = 'num'

        self.degenerate_eigenvalues = degenerate_eigenvalues
        self.n_sheets = n_sheets
        self.h = h
        self.hsymbols = None    # set when eigensystem_dipole_path is called
        self.hderiv = None   # set when eigensystem_dipole_path is called
        self.ana_e = ana_e
        self.ederivx = None
        self.ederivy = None
        self.ana_wf = ana_wf
        self.dipole_x = None
        self.dipole_y = None
        self.B = None

        self.efjit = None
        self.ederivfjit = None
        self.wfjit = None
        self.wfcjit = None
        self.dipolexjit = None
        self.dipoleyjit = None
        self.Bcurvjit = None

        self.n = h.shape[0]

        self.hfjit = None

        self.e_in_path = None   #set when eigensystem_dipole_path is called
        self.wf_in_path = None

        self.dipole_path_x = None   # set when eigensystem_dipole_path is called
        self.dipole_path_y = None

        self.dipole_in_path = None
        self.dipole_ortho = None

        self.dipole_derivative_in_path = None

        self.Bcurv_path = None

    def __hamiltonian_derivatives(self):
        return [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]

    def __energy_derivatives(self):
        """
        Calculate the derivative of the energy bands. Order is
        de[0]/dkx, de[0]/dky, de[1]/dkx, de[1]/dky
        """
        edx = []
        edy = []
        for e in self.ana_e:
            edx.append(sp.diff(e, self.kx))
            edy.append(sp.diff(e, self.ky))
        return edx, edy

    def __normalize_eigenvectors(self):

        for i in range(self.n):
            norm = 0
            for j in range(self.n):
                norm += sp.Abs(self.ana_wf[j, i])**2
            self.ana_wf[:, i] *= norm**(-1/2)

    def __fields(self):
        dUx = sp.diff(self.ana_wf, self.kx)
        dUy = sp.diff(self.ana_wf, self.ky)
        # Minus sign is the charge
        return -sp.I*(self.ana_wf.H) * dUx, -sp.I*(self.ana_wf.H) * dUy

    def __ana_berry_curvature(self):
        return sp.diff(self.dipole_x, self.kx) - sp.diff(self.dipole_y, self.kx)

    def __isZeeman(self, mag_strength):
        s_z = sp.Matrix([[1,0],[0,-1]])
        self.n = self.h.shape[0]
        return mag_strength*TensorProduct(s_z,sp.eye(int(self.n/2)))

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

        xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
        yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
        ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
        ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

        return xderivative, yderivative, ederivx, ederivy

    def __berry_curvature(self, path, P):

        epsilon = P.epsilon
        pathlen = path[:, 0].size

        dAydx = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)
        dAxdy = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)

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

        Ax_plusx, Ay_plusx = self.dipole_path(pathplusx, P)
        Ax_minusx, Ay_minusx = self.dipole_path(pathminusx, P)
        Ax_plusy, Ay_plusy = self.dipole_path(pathplusy, P)
        Ax_minusy, Ay_minusy = self.dipole_path(pathminusy, P)

        Ax_plus2x, Ay_plus2x = self.dipole_path(pathplus2x, P)
        Ax_minus2x, Ay_minus2x = self.dipole_path(pathminus2x, P)
        Ax_plus2y, Ay_plus2y = self.dipole_path(pathplus2y, P)
        Ax_minus2y, Ay_minus2y = self.dipole_path(pathminus2y, P)

        Ax_plus3x, Ay_plus3x = self.dipole_path(pathplus3x, P)
        Ax_minus3x, Ay_minus3x = self.dipole_path(pathminus3x, P)
        Ax_plus3y, Ay_plus3y = self.dipole_path(pathplus3y, P)
        Ax_minus3y, Ay_minus3y = self.dipole_path(pathminus3y, P)

        Ax_plus4x, Ay_plus4x = self.dipole_path(pathplus4x, P)
        Ax_minus4x, Ay_minus4x = self.dipole_path(pathminus4x, P)
        Ax_plus4y, Ay_plus4y = self.dipole_path(pathplus4y, P)
        Ax_minus4y, Ay_minus4y = self.dipole_path(pathminus4y, P)

        dAxdy = (1/280*(Ax_minus4y - Ax_plus4y) + 4/105*( Ax_plus3y - Ax_minus3y ) + 1/5*( Ax_minus2y - Ax_plus2y ) + 4/5*( Ax_plusy - Ax_minusy ) )/epsilon
        dAydx = (1/280*(Ay_minus4x - Ay_plus4x) + 4/105*( Ay_plus3x - Ay_minus3x ) + 1/5*( Ay_minus2x - Ay_plus2x ) + 4/5*( Ay_plusx - Ay_minusx ) )/epsilon

        Bcurv = np.zeros([pathlen, P.n], dtype=P.type_complex_np)

        for i in range(P.n):
            for i_k in range(pathlen):
                Bcurv[i_k, i] = dAxdy[i_k, i, i] - dAydx[i_k, i, i]

        return Bcurv

    def __dipole_derivative(self, path, P):

        epsilon = P.epsilon
        pathlen = path[:, 0].size

        dAydx = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)
        dAxdy = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)

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

        Ax_plusx, Ay_plusx = self.dipole_path(pathplusx, P)
        Ax_minusx, Ay_minusx = self.dipole_path(pathminusx, P)
        Ax_plusy, Ay_plusy = self.dipole_path(pathplusy, P)
        Ax_minusy, Ay_minusy = self.dipole_path(pathminusy, P)

        Ax_plus2x, Ay_plus2x = self.dipole_path(pathplus2x, P)
        Ax_minus2x, Ay_minus2x = self.dipole_path(pathminus2x, P)
        Ax_plus2y, Ay_plus2y = self.dipole_path(pathplus2y, P)
        Ax_minus2y, Ay_minus2y = self.dipole_path(pathminus2y, P)

        Ax_plus3x, Ay_plus3x = self.dipole_path(pathplus3x, P)
        Ax_minus3x, Ay_minus3x = self.dipole_path(pathminus3x, P)
        Ax_plus3y, Ay_plus3y = self.dipole_path(pathplus3y, P)
        Ax_minus3y, Ay_minus3y = self.dipole_path(pathminus3y, P)

        Ax_plus4x, Ay_plus4x = self.dipole_path(pathplus4x, P)
        Ax_minus4x, Ay_minus4x = self.dipole_path(pathminus4x, P)
        Ax_plus4y, Ay_plus4y = self.dipole_path(pathplus4y, P)
        Ax_minus4y, Ay_minus4y = self.dipole_path(pathminus4y, P)

        dAxdx = (1/280*(Ax_minus4x - Ax_plus4x) + 4/105*( Ax_plus3x - Ax_minus3x ) + 1/5*( Ax_minus2x - Ax_plus2x ) + 4/5*( Ax_plusx - Ax_minusx ) )/epsilon
        dAydy = (1/280*(Ay_minus4y - Ay_plus4y) + 4/105*( Ay_plus3y - Ay_minus3y ) + 1/5*( Ay_minus2y - Ay_plus2y ) + 4/5*( Ay_plusy - Ay_minusy ) )/epsilon
        dAxdy = (1/280*(Ax_minus4y - Ax_plus4y) + 4/105*( Ax_plus3y - Ax_minus3y ) + 1/5*( Ax_minus2y - Ax_plus2y ) + 4/5*( Ax_plusy - Ax_minusy ) )/epsilon
        dAydx = (1/280*(Ay_minus4x - Ay_plus4x) + 4/105*( Ay_plus3x - Ay_minus3x ) + 1/5*( Ay_minus2x - Ay_plus2x ) + 4/5*( Ay_plusx - Ay_minusx ) )/epsilon

        dipole_derivative = P.E_dir[0]*P.E_dir[0]*dAxdx + P.E_dir[0]*P.E_dir[1]*(dAxdy + dAydx) + P.E_dir[1]*P.E_dir[1]*dAydy

        return dipole_derivative

    def diagonalize_path(self, path, P):

        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        pathlen = path[:, 0].size
        e_path = np.empty([pathlen, P.n], dtype=P.type_real_np)
        wf_path = np.empty([pathlen, P.n, P.n], dtype=P.type_complex_np)
        h_in_path = evaluate_njit_matrix(self.hfjit, kx_in_path, ky_in_path)

        for i in range(pathlen):
            e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
            if self.degenerate_eigenvalues:
                for j in range(int(P.n/2)):
                    wf1 = np.copy(wf_buff[:, 2*j])
                    wf2 = np.copy(wf_buff[:, 2*j+1])
                    wf_buff[:, 2*j] *= wf2[P.n-2]
                    wf_buff[:, 2*j] -= wf1[P.n-2]*wf2
                    wf_buff[:, 2*j+1] *= wf1[P.n-1]
                    wf_buff[:, 2*j+1] -= wf2[P.n-1]*wf1
            wf_gauged_entry = np.copy(wf_buff[P.gidx, :])
            wf_buff[P.gidx, :] = np.abs(wf_gauged_entry)
            wf_buff[~(np.arange(np.size(wf_buff, axis=0)) == P.gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
            wf_path[i] = wf_buff

        return e_path, wf_path


    def dipole_path(self, path, P):

        pathlen = path[:, 0].size

        dx_path = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)
        dy_path = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)

        _buf, wf_path = self.diagonalize_path(path, P)
        dwfkx_path, dwfky_path, _buf, _buf = self.__derivative_path(path, P)

        for i in range(pathlen):
            dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
            dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

        return dx_path, dy_path


    def eigensystem_dipole_path(self, path, P):
        '''
            Dipole Elements of the Hamiltonian for a given path
        '''
        if P.Zeeman:
            self.h = self.h + self.__isZeeman(P.zeeman_strength)

        self.hsymbols = self.h.free_symbols
        self.hderiv = self.__hamiltonian_derivatives()

        if self.hfjit == None:
            self.hfjit = matrix_to_njit_functions(self.h, self.hsymbols, dtype=P.type_complex_np)
            self.hderivfjit = [matrix_to_njit_functions(hd, self.hsymbols, dtype=P.type_complex_np)
                            for hd in self.hderiv]
            
            if self.ana_e is not None:
                self.__normalize_eigenvectors()
                self.ederivx, self.ederivy = self.__energy_derivatives()
                self.dipole_x, self.dipole_y = self.__fields()
                self.B = self.__ana_berry_curvature()

                self.efjit = list_to_njit_functions(self.ana_e, self.hsymbols, dtype=P.type_complex_np)
                self.ederivxfjit = list_to_njit_functions(self.ederivx, self.hsymbols, dtype=P.type_complex_np)
                self.ederivyfjit = list_to_njit_functions(self.ederivy, self.hsymbols, dtype=P.type_complex_np)
                self.wfjit = matrix_to_njit_functions(self.ana_wf, self.hsymbols, dtype=P.type_complex_np)
                self.dipolexjit = matrix_to_njit_functions(self.dipole_x, self.hsymbols, dtype=P.type_complex_np)
                self.dipoleyjit = matrix_to_njit_functions(self.dipole_y, self.hsymbols, dtype=P.type_complex_np)
                self.Bcurvjit = matrix_to_njit_functions(self.B, self.hsymbols, dtype=P.type_complex_np)

        pathlen = path[:, 0].size

        if self.ana_e is not None:

            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]
            e_path = np.zeros([pathlen, P.n], dtype=P.type_real_np)
            edx_path = np.zeros([pathlen, P.n], dtype=P.type_real_np)
            edy_path = np.zeros([pathlen, P.n], dtype=P.type_real_np)
            
            for n, e in enumerate(self.efjit):
                e_path[:, n] = e(kx=kx_in_path, ky=ky_in_path)
            for n, e in enumerate(self.ederivxfjit):
                edx_path[:, n] = e(kx=kx_in_path, ky=ky_in_path)            
            for n, e in enumerate(self.ederivyfjit):
                edy_path[:, n] = e(kx=kx_in_path, ky=ky_in_path)

            wf_path = evaluate_njit_matrix(self.wfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.ederivx_path = edx_path
            self.ederivy_path = edy_path
        
        else:
            e_path, wf_path = self.diagonalize_path(path, P)
            _buf1, _buf2, self.ederivx_path, self.ederivy_path = self.__derivative_path(path, P)

        if P.dm_dynamics_method == 'semiclassics':
            self.dipole_path_x = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)
            self.dipole_path_y = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)

        elif self.ana_e is not None:
            self.dipole_path_x = evaluate_njit_matrix(self.dipolexjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.dipole_path_y = evaluate_njit_matrix(self.dipoleyjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        else:
            self.dipole_path_x, self.dipole_path_y = self.dipole_path(path, P)

        self.e_in_path = e_path
        self.wf_in_path = wf_path
        self.dipole_in_path = P.E_dir[0]*self.dipole_path_x + P.E_dir[1]*self.dipole_path_y
        self.dipole_ortho = P.E_ort[0]*self.dipole_path_x + P.E_ort[1]*self.dipole_path_y
        
        if self.ana_e is not None:
            B_path = np.zeros([pathlen, P.n], dtype=P.type_complex_np)
            for n, b in enumerate(self.Bcurvjit):
                B_path[:, n] = b[n](kx=kx_in_path, ky=ky_in_path)        
            self.Bcurv_path = B_path
        else:    
            self.Ax_path, self.Ay_path = self.dipole_path(path, P)
            self.Bcurv_path = self.__berry_curvature(path, P)

        if P.dm_dynamics_method == 'EEA':
            self.dipole_derivative_in_path = self.__dipole_derivative(path, P)
