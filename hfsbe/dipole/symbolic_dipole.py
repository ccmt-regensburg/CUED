import sympy as sp
import numpy as np

import hfsbe.check.symbolic_checks as symbolic_checks
import hfsbe.lattice as lattice
import hfsbe.utility as utility


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
            symbolic_checks.eigensystem(h, e, wf)

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
        self.fkwargs = None

    def __fields(self):
        dUx = sp.diff(self.U, self.kx)
        dUy = sp.diff(self.U, self.ky)
        return sp.I*self.U_h * dUx, sp.I*self.U_h * dUy

    def evaluate(self, kx, ky, b1=None, b2=None,
                 hamiltonian_radius=None, eps=10e-10, **fkwargs):
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
        hamiltonian_radius : float
            kspace radius of a disk where the hamiltonian is defined
        fkwargs :
            keyword arguments passed to the symbolic expression
        eps : float
            Threshold to identify Brillouin zone boundary points
        """
        hamr = hamiltonian_radius
        self.fkwargs = fkwargs

        if (b1 is None or b2 is None):
            # Evaluate all kpoints without BZ
            return kx, ky, self.Axf(kx=kx, ky=ky, **self.fkwargs), \
                self.Ayf(kx=kx, ky=ky, **self.fkwargs)
        else:
            # Add a BZ and cut off
            return self.__add_brillouin(kx, ky, b1, b2, hamr, eps)

    def __add_brillouin(self, kx, ky, b1, b2, hamr, eps):
        """
        Evaluate the dipole moments in a given Brillouin zone.
        """
        kxbz, kybz = lattice.in_brillouin(kx, ky, b1, b2, eps)

        if (hamr is None):
            # No hamiltonian region given -> defined in entire bz
            return kxbz, kybz, self.Axf(kx=kxbz, ky=kybz, **self.fkwargs), \
                self.Ayf(kx=kxbz, ky=kybz, **self.fkwargs)
        else:
            # Regular evaluation in circle
            Ax = np.empty(self.Ax.shape + (kxbz.size, ), dtype=np.complex)
            Ay = np.empty(self.Ay.shape + (kybz.size, ), dtype=np.complex)

            inci = kxbz**2 + kybz**2 <= hamr**2
            Ax[:, :, inci] = self.Axf(kx=kxbz[inci], ky=kybz[inci],
                                      **self.fkwargs)
            Ay[:, :, inci] = self.Ayf(kx=kxbz[inci], ky=kybz[inci],
                                      **self.fkwargs)

            # Interpolation out of circle
            outci = np.logical_not(inci)
            Axi, Ayi = self.__interpolate(kxbz[outci], kybz[outci],
                                          b1, b2, hamr)
            Ax[:, :, outci] = Axi
            Ay[:, :, outci] = Ayi

            return kxbz, kybz, Ax, Ay

    def __interpolate(self, kx, ky, b1, b2, hamr):
        """
        Interpolates everything outside of the Hamiltonian radius with
        a linear function, between the two circle ends
        """
        kmat = np.vstack((kx, ky))
        kx_b, ky_b = lattice.intersect_brillouin(kx, ky, b1, b2)

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


