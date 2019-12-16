import sympy as sp
import numpy as np

import hfsbe.check.symbolic_checks as symbolic_checks
from hfsbe.lattice import lattice.evaluate_in_brillouin as bzeval
import hfsbe.utility as utility


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

        if (test):
            symbolic_checks.eigensystem(h, e, wf)

        self.b1 = b1
        self.b2 = b2

        self.kx = sp.Symbol('kx')
        self.ky = sp.Symbol('ky')

        self.h = h
        self.e = e
        self.U = wf[0]
        self.U_h = wf[1]

        self.Ax, self.Ay = self.__fields()

        # Numpy function and function arguments
        self.Axf = utility.to_numpy_function(self.Ax)
        self.Ayf = utility.to_numpy_function(self.Ay)

    def __fields(self):
        dUx = sp.diff(self.U, self.kx)
        dUy = sp.diff(self.U, self.ky)
        return sp.I*self.U_h * dUx, sp.I*self.U_h * dUy

    def evaluate(self, kx, ky, hamiltonian_radius=None, eps=10e-10,
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
        hamiltonian_radius : float
            kspace radius of a disk where the hamiltonian is defined
        fkwargs :
            keyword arguments passed to the symbolic expression
        eps : float
            Threshold to identify Brillouin zone boundary points
        """

        hamr = hamiltonian_radius

        if (self.b1 is None or self.b2 is None):
            # Evaluate all kpoints without BZ
            return self.Axf(kx=kx, ky=ky, **fkwargs), \
                self.Ayf(kx=kx, ky=ky, **fkwargs)
        else:
            # Add a BZ and throw error if kx, ky is outside
            Ax_return = bzeval(self.Axf, kx, ky, self.b1, self.b2,
                               hamiltonian_radius=hamr, eps=eps,
                               **fkwargs)
            Ay_return = bzeval(self.Ayf, kx, ky, self.b1, self.b2,
                               hamiltonian_radius=hamr, eps=eps,
                               **fkwargs)

            return Ax_return, Ay_return

    def __interpolate(self, kx, ky, hamr):
        """
        Interpolates everything outside of the Hamiltonian radius with
        a linear function, between the two circle ends
        """
        kmat = np.vstack((kx, ky))
        kx_b, ky_b = lattice.intersect_brillouin(kx, ky, self.b1, self.b2)

        # interpolation length from circle edge over boundary to
        # circle edge
        ipl = 2*(np.linalg.norm(np.vstack((kx_b, ky_b)), axis=0) - hamr)
        knorm = np.linalg.norm(kmat, axis=0)

        # Point at circle, kvector intersection
        kmat_c = hamr*(kmat/knorm)
        pos = knorm - hamr
        # Aci_f circle evaluation forward facing to point
        # Aci_b circle evaulation on backside facing away from point
        Aci_f_x = self.Axf(kx=kmat_c[0], ky=kmat_c[1], **self.fkwargs)
        Aci_f_y = self.Ayf(kx=kmat_c[0], ky=kmat_c[1], **self.fkwargs)
        Aci_b_x = self.Axf(kx=-kmat_c[0], ky=-kmat_c[1], **self.fkwargs)
        Aci_b_y = self.Ayf(kx=-kmat_c[0], ky=-kmat_c[1], **self.fkwargs)

        Axi = (1-pos/ipl)*Aci_f_x + (pos/ipl)*Aci_b_x
        Ayi = (1-pos/ipl)*Aci_f_y + (pos/ipl)*Aci_b_y

        return Axi, Ayi
