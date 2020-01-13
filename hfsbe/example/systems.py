import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from hfsbe.brillouin import evaluate_scalar_field
from hfsbe.utility import list_to_numpy_functions


class TwoBandSystem():
    so = sp.Matrix([[1, 0], [0, 1]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    kx = sp.Symbol('kx')
    ky = sp.Symbol('ky')

    def __init__(self, ho, hx, hy, hz, b1=None, b2=None):
        """
        Generates the symbolic Hamiltonian, wave functions and
        energies.

        Parameters
        ----------
        ho, hx, hy, hz : Symbol
            Wheter to additionally return energy and wave function derivatives
        """
        self.b1 = b1
        self.b2 = b2

        self.ho = ho
        self.hx = hx
        self.hy = hy
        self.hz = hz

        # Get set when eigensystem is called
        self.e = None             # Symbolic energies
        self.ederiv = None        # Symbolic energy derivatives
        self.h = None             # Full symbolic Hamiltonian
        self.U = None             # Normalised eigenstates
        self.U_h = None           # Hermitian conjugate
        self.U_no_norm = None     # Unnormalised eigenstates
        self.U_h_no_norm = None   # Hermitian conjugate

        self.ef = None          # Energies as callable functions
        self.ederivf = None     # Energy derivatives as callable func.

        # Get set when evaluate_energy is called
        self.e_eval = None

    def __hamiltonian(self):
        return self.ho*self.so + self.hx*self.sx + self.hy*self.sy \
            + self.hz*self.sz

    def __wave_function(self, gidx=None):
        esoc = sp.sqrt(self.hx**2 + self.hy**2 + self.hz**2)

        if (gidx is None):
            wfv = sp.Matrix([-self.hx + sp.I*self.hy, self.hz + esoc])
            wfc = sp.Matrix([self.hz + esoc, self.hx + sp.I*self.hy])
            wfv_h = sp.Matrix([-self.hx - sp.I*self.hy, self.hz + esoc])
            wfc_h = sp.Matrix([self.hz + esoc, self.hx - sp.I*self.hy])
            normv = sp.sqrt(2*(esoc + self.hz)*esoc)
            normc = sp.sqrt(2*(esoc + self.hz)*esoc)
        elif (gidx == 0):
            wfv = sp.Matrix([1,
                             (self.hx+sp.I*self.hy)/(self.hz-esoc)])
            wfc = sp.Matrix([1,
                             (self.hx+sp.I*self.hy)/(self.hz+esoc)])
            wfv_h = sp.Matrix([1,
                               (self.hx-sp.I*self.hy)/(self.hz-esoc)])
            wfc_h = sp.Matrix([1,
                               (self.hx-sp.I*self.hy)/(self.hz-esoc)])
            normv = sp.sqrt(wfv_h.dot(wfv))
            normc = sp.sqrt(wfc_h.dot(wfc))
        elif (gidx == 1):
            wfv = sp.Matrix([-(self.hx-sp.I*self.hy)/(self.hz+esoc),
                             1])
            wfc = sp.Matrix([-(self.hx-sp.I*self.hy)/(self.hz-esoc),
                             1])
            wfv_h = sp.Matrix([-(self.hx+sp.I*self.hy)/(self.hz+esoc),
                               1])
            wfc_h = sp.Matrix([-(self.hx+sp.I*self.hy)/(self.hz-esoc),
                               1])
            normv = sp.sqrt(wfv_h.dot(wfv))
            normc = sp.sqrt(wfc_h.dot(wfc))

        U = (wfv/normv).row_join(wfc/normc)
        U_h = (wfv_h/normv).T.col_join((wfc_h/normc).T)

        U_no_norm = (wfv).row_join(wfc)
        U_h_no_norm = (wfv_h).T.col_join(wfc_h.T)

        return U, U_h, U_no_norm, U_h_no_norm

    def __energies(self):
        esoc = sp.sqrt(self.hx**2 + self.hy**2 + self.hz**2)
        return [self.ho - esoc, self.ho + esoc]

    def __ederiv(self, energies):
        """
        Calculate the derivative of the energy bands. Order is
        de[0]/dkx, de[0]/dky, de[1]/dkx, de[1]/dky
        """
        ed = []
        for e in energies:
            ed.append(sp.diff(e, self.kx))
            ed.append(sp.diff(e, self.ky))
        return ed

    def eigensystem(self, gidx=None):
        """
        Generic form of Hamiltonian, energies and wave functions in a two band
        Hamiltonian.

        Returns
        -------
        h : Symbol
            Hamiltonian of the system
        e : list of Symbol
            Valence and conduction band energies; in this order
        [U, U_h] : list of Symbol
            Valence and conduction band wave function; in this order
        ederiv : list of Symbol
            List of energy derivatives
        gidx : integer
            gauge index, index of the wave function entry where it is
            kept at 1. Can either be 0, 1 or None for the default
        """
        self.e = self.__energies()
        self.ederiv = self.__ederiv(self.e)
        self.h = self.__hamiltonian()
        self.U, self.U_h, self.U_no_norm, self.U_h_no_norm = \
            self.__wave_function(gidx=gidx)

        self.ef = list_to_numpy_functions(self.e)
        self.ederivf = list_to_numpy_functions(self.ederiv)

        return self.h, self.e, [self.U, self.U_h], self.ederiv

    def evaluate_energy(self, kx, ky, hamr=None,
                        eps=10e-10, **fkwargs):
        # Evaluate all kpoints without BZ
        self.e_eval = []
        if (self.b1 is None or self.b2 is None):
            for ef in self.ef:
                self.e_eval.append(ef(kx=kx, ky=ky, **fkwargs))
            return self.e_eval

        # Add a BZ and throw error if kx, ky is outside
        for ef in self.ef:
            buff = evaluate_scalar_field(ef, kx, ky, self.b1, self.b2,
                                         hamr=hamr, eps=eps,
                                         **fkwargs)
            self.e_eval.append(buff)

        return self.e_eval

    def evaluate_ederivative(self, kx, ky, hamr=None,
                             eps=10e-10, **fkwargs):
        ederivat = []
        # Evaluate all kpoints without BZ
        if (self.b1 is None or self.b2 is None):
            for ederivf in self.ederivf:
                ederivat.append(ederivf(kx=kx, ky=ky, **fkwargs))
            return ederivat

        for ederivf in self.ederivf:
            buff = evaluate_scalar_field(ederivf, kx, ky, self.b1, self.b2,
                                         hamr, eps=eps, **fkwargs)
            ederivat.append(buff)

        return ederivat

    def plot_energies_3d(self, kx, ky, title="Energies"):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(kx, ky, self.e_eval[0])

        plt.show()


class Haldane(TwoBandSystem):
    """
    Haldane model
    """
    def __init__(self, b1=None, b2=None):
        t1 = sp.Symbol('t1')
        t2 = sp.Symbol('t2')
        m = sp.Symbol('m')
        phi = sp.Symbol('phi')

        a1 = self.kx
        a2 = -1/2 * self.kx + sp.sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sp.sqrt(3)/2 * self.ky

        b1 = sp.sqrt(3) * self.ky
        b2 = -3/2 * self.kx - sp.sqrt(3)/2 * self.ky
        b3 = 3/2 * self.kx - sp.sqrt(3)/2 * self.ky

        ho = 2*t2*sp.cos(phi)*(sp.cos(b1)+sp.cos(b2)+sp.cos(b3))
        hx = t1*(sp.cos(a1)+sp.cos(a2)+sp.cos(a3))
        hy = t1*(sp.sin(a1)+sp.sin(a2)+sp.sin(a3))
        hz = m - 2*t2*sp.sin(phi)*(sp.sin(b1)+sp.sin(b2)+sp.sin(b3))

        super().__init__(ho, hx, hy, hz, b1=b1, b2=b2)


class BiTe(TwoBandSystem):
    """
    Bismuth Telluride topological insulator model
    """
    def __init__(self, C0=sp.Symbol('C0'), C2=sp.Symbol('C2'),
                 A=sp.Symbol('A'), R=sp.Symbol('R'), vf=sp.Symbol('vf'),
                 kcut=None, b1=None, b2=None, default_params=False):
        if (default_params):
            A, R, C0, C2 = self.__set_default_params()

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2)

        if (kcut is not None):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor

        hz += vf*sp.sqrt(self.kx**2 + self.ky**2)

        super().__init__(ho, hx, hy, hz, b1=b1, b2=b2)

    def __set_default_params(self):
        """
        Default BiTe system parameters in atomic units
        """
        A = 0.1974
        R = 11.06
        C0 = -0.008269
        C2 = 6.5242
        return A, R, C0, C2


class Graphene(TwoBandSystem):
    """
    Graphene model
    """
    def __init__(self, t=sp.Symbol('t'), b1=None, b2=None):
        a1 = self.kx
        a2 = -1/2 * self.kx + sp.sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sp.sqrt(3)/2 * self.ky

        ho = 0
        hx = t*(sp.cos(a1)+sp.cos(a2)+sp.cos(a3))
        hy = t*(sp.sin(a1)+sp.sin(a2)+sp.sin(a3))
        hz = 0

        super().__init__(ho, hx, hy, hz, b1=b1, b2=b2)


class QWZ(TwoBandSystem):
    """
    Qi-Wu-Zhang model of a 2D Chern insulator
    """
    def __init__(self, order=sp.oo, b1=None, b2=None):
        n = order+1
        m = sp.Symbol('m')

        ho = 0
        if order == sp.oo:
            hx = sp.sin(self.kx)
            hy = sp.sin(self.ky)
            hz = m - sp.cos(self.kx) - sp.cos(self.ky)
        else:
            hx = sp.sin(self.kx).series(n=n).removeO()
            hy = sp.sin(self.ky).series(n=n).removeO()
            hz = m - sp.cos(self.kx).series(n=n).removeO()\
                - sp.cos(self.ky).series(n=n).removeO()

        super().__init__(ho, hx, hy, hz, b1=b1, b2=b2)


class Dirac(TwoBandSystem):
    """
    Generic Dirac cone Hamiltonian
    """
    def __init__(self, m=sp.Symbol('m'), b1=None, b2=None):

        ho = 0
        hx = self.kx
        hy = self.ky
        hz = m

        super().__init__(ho, hx, hy, hz, b1=b1, b2=b2)
