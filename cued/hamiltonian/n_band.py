import numpy as np
import sympy as sp

from cued.utility import list_to_njit_functions, matrix_to_njit_functions


class NBandHamiltonianSystem():

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    def __init__(self, h):
    
        self.h = h
        self.hsymbols = self.h.free_symbols
        self.hderiv = self.__hamiltonian_derivatives()

        self.hfjit = matrix_to_njit_functions(self.h, self.hsymbols)
        self.hderivfjit = [matrix_to_njit_functions(hd, self.hsymbols)
                           for hd in self.hderiv]
                           
    def __hamiltonian_derivatives(self):
        return [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]

class BiTe3(NBandHamiltonianSystem):
    '''
        Artificial 3Band model with Dirac cone for first two bands, zero else
    '''
    
    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0):

        so = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        sy = sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]])
        sz = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2) + mz

        if (not np.isclose(kcut, 0)):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor
        h = ho*so + hx*sx + hy*sy + hz*sz

        super().__init__(h)
