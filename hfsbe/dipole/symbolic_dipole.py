import sympy as sp
import numpy as np

import hfsbe.check.symbolic_checks as sck


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

    def evaluate(self, kx, ky, b1=None, b2=None,
                 interpolation_ratio=1.0, **kwargs):
        """
        Transforms the symbolic expression for the
        berry connection/dipole moment matrix to an expression
        that is numerically evaluated.
        If the reciprocal lattice vectors are given it creates a
        Brillouin zone around the symbolic Hamiltonian. Values outside
        of that zone are returned as np.nan.
        The interpolation ratio (ipr) determines the part of the Brillouin
        zone the symbolic Hamiltonian can be defined on. Outside of
        this region up to the Brillouin zone boundaries the
        dipole moments will be interpolated by constant values
        given at the edge of the small zone given by ipr*b1 + ipr*b2

        Parameters:
        kx, ky : np.ndarray
            array of all point combinations
        b1, b2 : np.ndarray
            reciprocal lattice vector
        interpolation_ratio : float
            percentile portion of reciprocal lattice vectors
        kwargs :
            keyword arguments passed to the symbolic expression
        """
        ipr = interpolation_ratio
        Axf = to_numpy_function(self.Ax)
        Ayf = to_numpy_function(self.Ay)

        if (b1 is None or b2 is None):
            return Axf(kx=kx, ky=ky, **kwargs), \
                Ayf(kx=kx, ky=ky, **kwargs)
        else:
            return self.__add_brillouin_zone(b1, b2, ipr)

    def __add_brillouin_zone(self, b1, b2, ip):
        """
        Add the brillouin zone to the symbolic dipole moments and
        return numpy callable function.
        """
        Ax_n = to_numpy_function(self.Ax)
        Ay_n = to_numpy_function(self.Ay)
        
        def brillouin_zone_function(**kwargs):
            kmat = np.vstack(kwargs.pop('kx'), kwargs.pop('ky'))
            b1.dot(kmat)
            b2.dot(kmat)

            
            Ax_n(kwargs)
            Ay_n(kwargs)


def to_numpy_function(sf):
    """
    Converts a simple sympy function/matrix to a function/matrix
    callable by numpy
    """

    return sp.lambdify(sf.free_symbols, sf, "numpy")


def list_to_numpy_functions(sf):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """

    return [to_numpy_function(sfn) for sfn in sf]
