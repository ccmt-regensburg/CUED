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
                 interpolation_ratio=1.0, eps=10e-10, **kwargs):
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
        eps : float
            Threshold to identify Brillouin zone boundary points
        """
        ipr = interpolation_ratio

        if (b1 is None or b2 is None):
            Axf = to_numpy_function(self.Ax)
            Ayf = to_numpy_function(self.Ay)
            return kx, ky, Axf(kx=kx, ky=ky, **kwargs), \
                Ayf(kx=kx, ky=ky, **kwargs)
        else:
            kmat = np.vstack((kx, ky))
            return self.__add_brillouin(kmat, b1, b2, ipr, eps, **kwargs)

    def __add_brillouin_zone(self, kmat, b1, b2, ipr, eps, **kwargs):
        """
        Evaluate the dipole moments in a given Brillouin zone.
        """
        Axf = to_numpy_function(self.Ax)
        Ayf = to_numpy_function(self.Ay)
        is_inbz = self.__check_zone(kmat, b1, b2, eps)
        kx = kmat[0, is_inbz]
        ky = kmat[1, is_inbz]
        if (ipr == 1.0):
            return kx, ky, Axf(kx=kx, ky=ky, **kwargs), \
                Ayf(kx=kx, ky=ky, **kwargs)
        else:
            is_inipr = self.__check_zone(kmat, ipr*b1, ipr*b2, eps)


    def __check_zone(self, kmat, b1, b2, eps):
        """
        Checks if a collection of k-points is inside a zone determined
        by the reciprocal lattice vectors b1, b2.
        """
        # projections of kpoints on reciprocal lattice vectors
        a1 = b1.dot(kmat)/(np.linalg.norm(b1)**2)
        a2 = b2.dot(kmat)/(np.linalg.norm(b2)**2)

        # smaller than half reciprocal lattice vector
        is_less_a1 = np.abs(a1) <= 0.5 + eps
        is_less_a2 = np.abs(a2) <= 0.5 + eps
        is_less_abs = np.abs(a1+a2) <= 0.5 + eps

        is_inzone = np.logical_and(is_less_a1, is_less_a2)
        is_inzone = np.logical_and(is_inzone, is_less_abs)
        return is_inzone


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
