import sympy as sp

import hfsbe.check.symbolic_checks as sck
import hfsbe.dipole.interface as itf


class SymbolicDipole():
    """
    This class constructs the dipole moment functions from a given symbolic
    Hamiltonian and wave function. It also performs checks on the input
    wave function to guarantee orthogonality and normalisation.

    """

    def __init__(self, h, e, wf, test=False):
        """
        Parameters
        ----------
        h : Symbol
            Hamiltonian of the system
        e : np.ndarray of Symbol
            Band energies of the system
        wf : np.ndarray of Symbol
            Wave functions, columns: bands, rows: wf and complex conjugate
        test : bool
            Wheter to perform a orthonormality and eigensystem test
        """

        if (test):
            sck.eigensystem(h, e, wf)

        self.kx = sp.Symbol('kx')
        self.ky = sp.Symbol('ky')

        self.h = h
        self.e = e
        self.U = wf[0]
        self.U_h = wf[1]

        self.Ax, self.Ay = self.__fields()

    def __fields(self):
        dUx = sp.diff(self.U, self.kx)
        dUy = sp.diff(self.U, self.ky)
        return self.U_h * dUx, self.U_h * dUy

    @property
    def Ax_numeric(self):
        itf.to_numpy_function(self.Ax)

    @property
    def Ay_numeric(self):
        itf.to_numpy_function(self.Ay) 
