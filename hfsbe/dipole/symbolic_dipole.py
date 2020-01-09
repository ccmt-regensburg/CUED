import sympy as sp
import numpy as np

import hfsbe.check.symbolic_checks as symbolic_checks
from hfsbe.brillouin import evaluate_matrix_field as evaldip
from hfsbe.utility import to_numpy_function


class SymbolicDipole():
    """
    This class constructs the dipole moment functions from a given symbolic
    Hamiltonian and wave function. It also performs checks on the input
    wave function to guarantee orthogonality and normalisation.

    """

    def __init__(self, h, e, wf, test=False, b1=None, b2=None):
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
        b1, b2 : np.ndarray
            reciprocal lattice vector
        """
        self.b1 = b1
        self.b2 = b2

        if (test):
            symbolic_checks.eigensystem(h, e, wf)

        self.kx = sp.Symbol('kx')
        self.ky = sp.Symbol('ky')

        self.h = h
        self.e = e
        self.U = wf[0]
        self.U_h = wf[1]

        self.Ax, self.Ay = self.__fields()

        # Numpy function and function arguments
        self.Axf = to_numpy_function(self.Ax)
        self.Ayf = to_numpy_function(self.Ay)

    def __fields(self):
        dUx = sp.diff(self.U, self.kx)
        dUy = sp.diff(self.U, self.ky)
        return sp.I*self.U_h * dUx, sp.I*self.U_h * dUy

    def evaluate(self, kx, ky, hamr=None, eps=10e-10,
                 **fkwargs):
        """
        Transforms the symbolic expression for the
        berry connection/dipole moment matrix to an expression
        that is numerically evaluated.
        If the reciprocal lattice vectors are given it creates a
        Brillouin zone around the symbolic Hamiltonian. Values outside
        of that zone are returned as np.nan.
        The hamiltonian_radius determines the part of the Brillouin
        zone the symbolic Hamiltonian can be defined on. Outside of
        this region up to the Brillouin zone boundaries the
        dipole moments will be interpolated by constant values
        given at the edge of the small zone given by h_r*b1 + h_r*b2

        Parameters:
        kx, ky : np.ndarray
            array of all point combinations
        hamr : float
            percentace of reciprocal lattice vectors where
            hamiltonian is defined
        fkwargs :
            keyword arguments passed to the symbolic expression
        eps : float
            Threshold to identify Brillouin zone boundary points
        """
        # Evaluate all kpoints without BZ
        if (self.b1 is None or self.b2 is None):
            return self.Axf(kx=kx, ky=ky, **fkwargs), \
                self.Ayf(kx=kx, ky=ky, **fkwargs)

        # Add a BZ and throw error if kx, ky is outside
        Ax_return = evaldip(self.Axf, kx, ky, self.b1, self.b2,
                            hamr=hamr, eps=eps,
                            **fkwargs)
        Ay_return = evaldip(self.Ayf, kx, ky, self.b1, self.b2,
                            hamr=hamr, eps=eps,
                            **fkwargs)

        return Ax_return, Ay_return
