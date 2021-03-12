import numpy as np
import sympy as sp

from cued.utility import evaluate_njit_matrix, list_to_njit_functions, matrix_to_njit_functions

class NBandBandstructureDipoleSystem():

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    def __init__(self, e, prefac_x, prefac_y, n, flag):

        self.prefac_x = prefac_x
        self.prefac_y = prefac_y
        self.e = e
        self.n = n  
        self.flag = flag
        
        self.freesymbols = set()

        for i in range(self.n):
            self.freesymbols.update(e[i].free_symbols)

        self.dkxe, self.dkye = self.energy_derivative()
        self.dipole_x, self.dipole_y = self.dipole_elements()
        self.matrix_element_x, self.matrix_element_y = self.matrix_elements()

        self.efjit = list_to_njit_functions(self.e, self.freesymbols)
        
        self.dkxejit = list_to_njit_functions(self.dkxe, self.freesymbols)
        self.dkyejit = list_to_njit_functions(self.dkye, self.freesymbols)
        
        self.dipole_xfjit = matrix_to_njit_functions(self.dipole_x, self.freesymbols)
        self.dipole_yfjit = matrix_to_njit_functions(self.dipole_y, self.freesymbols)

        self.melxjit = matrix_to_njit_functions(self.matrix_element_x, self.freesymbols)
        self.melyjit = matrix_to_njit_functions(self.matrix_element_y, self.freesymbols)

        self.e_in_path = None   #set when eigensystem_dipole_path is called

        self.dipole_path_x = None   
        self.dipole_path_y = None
                           
        self.dipole_in_path = None
        self.dipole_ortho = None

    def energy_derivative(self):
        dkxe = sp.zeros(self.n)
        dkye = sp.zeros(self.n)
        for i, en in enumerate(self.e):
            dkxe[i] = sp.diff(en, self.kx)
            dkye[i] = sp.diff(en, self.ky)
        return dkxe, dkye

    def dipole_elements(self):
        
        if self.flag == 'dipole':
            dipole_x = self.prefac_x
            dipole_y = self.prefac_y
        else:
            dipole_x = sp.zeros(self.n, self.n)
            dipole_y = sp.zeros(self.n, self.n)
            
            for i in range(self.n):        
                for j in range(self.n):
                    if i == j:              #diagonal elements are zero
                        dipole_x[i, j] = 0
                        dipole_y[i, j] = 0
                    else:                   #offdiagonal elements from formula
                        if self.flag == 'd0':
                            e0i = self.ejit[i](kx = 0, ky = 0)
                            e0j = self.ejit[j](kx = 0, ky = 0)
                            dipole_x[i, j] = self.prefac_x[i, j] * ( e0j - e0i ) / ( self.e[j] - self.e[i] )
                            dipole_y[i, j] = self.prefac_y[i, j] * ( e0j - e0i ) / ( self.e[j] - self.e[i] )
                        if self.flag == 'prefac':
                            dipole_x[i, j] = self.prefac_x[i, j] / ( self.e[j] - self.e[i] )
                            dipole_y[i, j] = self.prefac_y[i, j] / ( self.e[j] - self.e[i] )
            
        return dipole_x, dipole_y

    def eigensystem_dipole_path(self, path, E_dir, P):

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        pathlen = path[:,0].size
        self.e_in_path = np.zeros([pathlen, P.n], dtype=P.type_real_np)

        E_ort = np.array([E_dir[1], -E_dir[0]])   

        self.dipole_path_x = evaluate_njit_matrix(self.dipole_xfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
        self.dipole_path_y = evaluate_njit_matrix(self.dipole_yfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        for n, e in enumerate(self.efjit):
            self.e_in_path[:, n] = e(kx=kx_in_path, ky=ky_in_path)

        self.dipole_in_path = E_dir[0]*self.dipole_path_x + E_dir[1]*self.dipole_path_y
        self.dipole_ortho = E_ort[0]*self.dipole_path_x + E_ort[1]*self.dipole_path_y        


    def matrix_elements(self):

        matrix_element_x = sp.zeros(self.n, self.n)
        matrix_element_y = sp.zeros(self.n, self.n)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    matrix_element_x[i, j] = self.dkxe[i]
                    matrix_element_y[i, j] = self.dkye[i]
                else:
                    matrix_element_x[i, j] = sp.I * self.dipole_x[i, j] * ( self.e[j] - self.e[i] )
                    #melx_buf.append(sp.I * self.prefac_x)
                    matrix_element_y[i, j] = sp.I * self.dipole_y[i, j] * ( self.e[j] - self.e[i] )
                    #mely_buf.append(sp.I * self.prefac_y)
        
        return matrix_element_x, matrix_element_y
    
###############################################################################################

class ExampleTwoBand(NBandBandstructureDipoleSystem):

    def __init__(self, 
                a = sp.Symbol('a', real=True), 
                prefac_x = sp.Symbol('prefac_x'), 
                prefac_y = sp.Symbol('prefac_y') , flag = None):


        ev = sp.cos(a*self.kx + self.ky)
        ec = sp.sin(a*self.ky)

        e = [ev, ec]
        n = 2 

        super().__init__(e, prefac_x, prefac_y, n, flag)

class DiracBandstructure(NBandBandstructureDipoleSystem):

    def __init__(self, vF = sp.Symbol('vF', real=True),
                prefac_x = sp.Symbol('prefac_x'),
                prefac_y = sp.Symbol('prefac_y'), flag = None):
                
        ev = - vF * sp.sqrt( self.kx**2 + self.ky**2 )
        ec =   vF * sp.sqrt( self.kx**2 + self.ky**2 )

        e = [ev, ec]
        n = 2

        super().__init__(e, prefac_x, prefac_y, n, flag)


class ExampleThreeBand(NBandBandstructureDipoleSystem):

    def __init__(self, 
                a = sp.Symbol('a', real=True), 
                prefac_x = sp.Symbol('prefac_x'), 
                prefac_y = sp.Symbol('prefac_y') , flag = None):

        e = []

        e1 = sp.cos(a*self.kx)
        e.append(e1)
        e2 = sp.sin(a*self.ky)
        e.append(e2)
        e3 = sp.cos(2*a*self.kx+self.ky)
        e.append(e3)
        n = 3

        super().__init__(e, prefac_x, prefac_y, n, flag)
