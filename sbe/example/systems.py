import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from sbe.utility import to_numpy_function, list_to_numpy_functions, \
    list_to_njit_functions, matrix_to_njit_functions

plt.rcParams['figure.figsize'] = [12, 15]
plt.rcParams['text.usetex'] = True


class TwoBandSystem():
    so = sp.Matrix([[1, 0], [0, 1]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    m_zee_x = sp.Symbol('m_zee_x', real=True)
    m_zee_y = sp.Symbol('m_zee_y', real=True)
    m_zee_z = sp.Symbol('m_zee_z', real=True)

    def __init__(self, ho, hx, hy, hz):
        """
        Generates the symbolic Hamiltonian, wave functions and
        energies.

        Parameters
        ----------
        ho, hx, hy, hz : Symbol
            Wheter to additionally return energy and wave function derivatives
        """

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

        self.ef = None            # Energies as callable functions
        self.ederivf = None       # Energy derivatives as callable func.

        self.efjit = None         # Energies as callable jit functions
        self.ederivfjit = None     # Energy derivatives as callable jit func

        # Get set when evaluate_energy is called
        self.e_eval = None
        self.ederiv_eval = None

    def __hamiltonian(self):
        return self.ho*self.so + self.hx*self.sx + self.hy*self.sy \
            + self.hz*self.sz

    def __hamiltonian_deriv(self):
        return [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]

    def __wave_function(self, gidx=None):
        esoc = sp.sqrt(self.hx**2 + self.hy**2 + self.hz**2)

        if gidx is None:
            wfv = sp.Matrix([-self.hx + sp.I*self.hy, self.hz + esoc])
            wfc = sp.Matrix([self.hz + esoc, self.hx + sp.I*self.hy])
            wfv_h = sp.Matrix([-self.hx - sp.I*self.hy, self.hz + esoc])
            wfc_h = sp.Matrix([self.hz + esoc, self.hx - sp.I*self.hy])
            normv = sp.sqrt(2*(esoc + self.hz)*esoc)
            normc = sp.sqrt(2*(esoc + self.hz)*esoc)
        elif 0 <= gidx <= 1:
            wfv_up = sp.Matrix([self.hz-esoc,
                                (self.hx+sp.I*self.hy)])
            wfc_up = sp.Matrix([self.hz+esoc,
                                (self.hx+sp.I*self.hy)])
            wfv_up_h = sp.Matrix([self.hz-esoc,
                                  (self.hx-sp.I*self.hy)])
            wfc_up_h = sp.Matrix([self.hz+esoc,
                                  (self.hx-sp.I*self.hy)])

            wfv_do = sp.Matrix([-self.hx+sp.I*self.hy,
                                self.hz+esoc])
            wfc_do = sp.Matrix([-self.hx+sp.I*self.hy,
                                self.hz-esoc])
            wfv_do_h = sp.Matrix([-self.hx-sp.I*self.hy,
                                  self.hz+esoc])
            wfc_do_h = sp.Matrix([-self.hx-sp.I*self.hy,
                                  self.hz-esoc])

            wfv = (1-gidx)*wfv_up + gidx*wfv_do
            wfc = (1-gidx)*wfc_up + gidx*wfc_do
            wfv_h = (1-gidx)*wfv_up_h + gidx*wfv_do_h
            wfc_h = (1-gidx)*wfc_up_h + gidx*wfc_do_h
            normv = sp.sqrt(wfv_h.dot(wfv))
            normc = sp.sqrt(wfc_h.dot(wfc))
        else:
            raise RuntimeError("gidx needs to be between 0 and 1 or None")

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

        Parameters
        ----------
        gidx : integer
            gauge index, index of the wave function entry where it is
            kept at 1. Can either be 0, 1 or None for the default

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

        """
        self.e = self.__energies()
        self.ederiv = self.__ederiv(self.e)
        self.h = self.__hamiltonian()
        self.hderiv = self.__hamiltonian_deriv()
        self.U, self.U_h, self.U_no_norm, self.U_h_no_norm = \
            self.__wave_function(gidx=gidx)

        hsymbols = self.h.free_symbols
        # Populate callable Hamiltonian, Hamiltonian derivative functions
        self.hf = to_numpy_function(self.h)
        self.hderivf = list_to_numpy_functions(self.hderiv)

        # Populate callable wave function functions
        self.Uf = to_numpy_function(self.U)
        self.Uf_h = to_numpy_function(self.U_h)

        self.ef = list_to_numpy_functions(self.e)
        self.ederivf = list_to_numpy_functions(self.ederiv)

        # All jitted functions
        self.hfjit = matrix_to_njit_functions(self.h, hsymbols)
        self.hderivfjit = [matrix_to_njit_functions(hd, hsymbols)
                           for hd in self.hderiv]
        self.Ujit = matrix_to_njit_functions(self.U, hsymbols)
        self.Ujit_h = matrix_to_njit_functions(self.U_h, hsymbols)

        self.efjit = list_to_njit_functions(self.e, hsymbols)
        self.ederivfjit = list_to_njit_functions(self.ederiv, hsymbols)

        return self.h, self.e, [self.U, self.U_h], self.ederiv

    def evaluate_energy(self, kx, ky, **fkwargs):
        # Evaluate all kpoints without BZ
        self.e_eval = []

        for ef in self.ef:
            self.e_eval.append(ef(kx=kx, ky=ky, **fkwargs))
        return self.e_eval

    def evaluate_ederivative(self, kx, ky, **fkwargs):
        self.ederiv_eval = []
        # Evaluate all kpoints without BZ
        if (self.b1 is None or self.b2 is None):
            for ederivf in self.ederivf:
                self.ederiv_eval.append(ederivf(kx=kx, ky=ky, **fkwargs))
            return self.ederiv_eval

    def plot_bands_3d(self, kx, ky, title="Energies"):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(kx, ky.T, self.e_eval[0])
        ax.plot_trisurf(kx, ky.T, self.e_eval[1])

        plt.title(title)
        plt.show()

    def plot_bands_scatter(self, kx, ky, vidx=0, cidx=1,
                           title=None, vname=None, cname=None,
                           xlabel=None, ylabel=None, clabel=None):
        """
        Plot the specified Bands.

        Parameters:
        kx, ky : np.ndarray
            array of all point combinations (same as evaluate)
        vidx, cidx : int
            Index of the first and second band to evaluate
        title : string
            Title of the plot
        vname, cname : string or int
            Index of names of the valence and conduction band
        xlabel, ylabel, clabel: string
            Label of x, y- axis and colorbar
        """
        if (title is None):
            title = "Band structure"
        if (vname is None):
            vname = vidx
        if (cname is None):
            cname = cidx
        if (xlabel is None):
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if (ylabel is None):
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if (clabel is None):
            clabel = r'Energy $[\mathrm{a.u.}]$'

        E = self.e_eval

        if (E is None):
            raise RuntimeError("Error: The curvature fields first need to"
                               " be evaluated on a kgrid to plot them. "
                               " Call evaluate before plotting.")

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title, fontsize=16)

        valence = ax[0].scatter(kx, ky, s=2, c=E[vidx], cmap="cool")
        ax[0].set_title(r"$E_{" + str(vname) + "}$")
        ax[0].axis('equal')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0], label=clabel)

        conduct = ax[1].scatter(kx, ky, s=2, c=E[cidx], cmap="cool")
        ax[1].set_title(r"$E_{" + str(cname) + "}$")
        ax[1].axis('equal')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[1], label=clabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_bands_contour(self, kx, ky, levels=10, vidx=0, cidx=1,
                           title=None, vname=None, cname=None,
                           xlabel=None, ylabel=None, clabel=None):
        """
        Plot the specified Bands.

        Parameters:
        kx, ky : np.ndarray
            array of all point combinations (same as evaluate)
        vidx, cidx : int
            Index of the first and second band to evaluate
        title : string
            Title of the plot
        vname, cname : string or int
            Index of names of the valence and conduction band
        xlabel, ylabel, clabel: string
            Label of x, y- axis and colorbar
        """
        if title is None:
            title = "Band structure"
        if vname is None:
            vname = vidx
        if cname is None:
            cname = cidx
        if xlabel is None:
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if ylabel is None:
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if clabel is None:
            clabel = r'Energy $[\mathrm{a.u.}]$'

        E = self.e_eval

        if E is None:
            raise RuntimeError("Error: The curvature fields first need to"
                               " be evaluated on a kgrid to plot them. "
                               " Call evaluate before plotting.")

        # Countour plot needs data in matrix form
        dim = int(np.sqrt(kx.size))
        kx = kx.reshape(dim, dim)
        ky = ky.reshape(dim, dim)
        ev = E[vidx].reshape(dim, dim)
        ec = E[cidx].reshape(dim, dim)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title, fontsize=16)

        cv = ax[0].contour(kx, ky, ev, levels=levels)
        plt.clabel(cv, inline=False, fontsize=10)
        ax[0].set_title(r"$E_{" + str(vname) + "}$")
        ax[0].axis('equal')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)

        cc = ax[1].contour(kx, ky, ec, levels=levels)
        plt.clabel(cc, inline=False, fontsize=10)
        ax[1].set_title(r"$E_{" + str(cname) + "}$")
        ax[1].axis('equal')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_bands_derivative(self, kx, ky, vidx=0, cidx=1,
                              title=None, vname=None, cname=None,
                              xlabel=None, ylabel=None, clabel=None):
        if title is None:
            title = "Energy derivatives"
        if vname is None:
            vname = vidx
        if cname is None:
            cname = cidx
        if xlabel is None:
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if ylabel is None:
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if clabel is None:
            clabel = r'$[\mathrm{a.u.}]$'

        devx = self.ederiv_eval[0]
        devy = self.ederiv_eval[1]
        decx = self.ederiv_eval[2]
        decy = self.ederiv_eval[3]

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title, fontsize=16)

        norm_valence = np.sqrt(devx**2 + devy**2)
        devx /= norm_valence
        devy /= norm_valence

        valence = ax[0].quiver(kx, ky, devx, devy, norm_valence,
                               angles='xy', cmap='cool')
        current_name = r"$\mathbf{\nabla}_k \epsilon_" + str(vname) + "$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[0].set_title(current_name)
        ax[0].axis('equal')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0], label=current_abs_name + clabel)

        norm_conduct = np.sqrt(decx**2 + decy**2)
        decx /= norm_conduct
        decy /= norm_conduct
        conduct = ax[1].quiver(kx, ky, decx, decy, norm_conduct,
                               angles='xy', cmap='cool')
        current_name = r"$\mathbf{\nabla}_k \epsilon_" + str(cname) + "$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[1].set_title(current_name)
        ax[1].axis('equal')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[1], label=current_abs_name + clabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


class Haldane(TwoBandSystem):
    """
    Haldane model
    """

    def __init__(self, t1=sp.Symbol('t1'), t2=sp.Symbol('t2'),
                 m=sp.Symbol('m'), phi=sp.Symbol('phi')):

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

        super().__init__(ho, hx, hy, hz)


class two_site_semiconductor(TwoBandSystem):
    """
    k-dependent Hamiltonian of 1d tight-binding chain with two sites in the unit cell
    """

    def __init__(self, 
                 lattice_const=sp.Symbol('lattice_const', real=True),
                 hopping=sp.Symbol('hopping', real=True),
                 onsite_energy_difference=sp.Symbol('onsite_energy_difference', real=True)) :

        ho = 0.000001*self.ky
        hx = hopping * (1 + sp.cos (self.kx * lattice_const))
        hy = - hopping * sp.sin (self.kx * lattice_const)
        hz = onsite_energy_difference/2

        super().__init__(ho, hx, hy, hz)



class BiTe(TwoBandSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0, zeeman=False):

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2) + mz

        if (not np.isclose(kcut, 0)):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor

        if (zeeman):
            hx -= self.m_zee_x
            hy -= self.m_zee_y
            hz -= self.m_zee_z

        super().__init__(ho, hx, hy, hz)


class BiTePeriodic(TwoBandSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, A=sp.Symbol('A', real=True),
                 C2=sp.Symbol('C2', real=True),
                 R=sp.Symbol('R', real=True),
                 a=sp.Symbol('a', real=True),
                 mw=1, order=4,
                 zeeman=False):

        kx = self.kx
        ky = self.ky

        sqr = sp.sqrt(3)
        pre = sp.Rational(1, 2)*a

        K1 = pre*(kx + sqr*ky)
        K2 = -pre*2*kx
        K3 = pre*(kx - sqr*ky)

        ho = (4/3)*(C2/a**2)*(-sp.cos(K1) - sp.cos(K2) - sp.cos(K3) + 3)
        hx = (1/sqr)*(A/a)*(sp.sin(K1) - sp.sin(K3))
        hy = (1/3)*(A/a)*(2*sp.sin(K2) - sp.sin(K1) - sp.sin(K3))
        hz = 16*(R/a**3)*(sp.sin(K1) + sp.sin(K2) + sp.sin(K3))
        # Wilson mass term
        hz += mw*8*(R/a**3)*3*sqr*4**(-order) \
            * (-sp.cos(K1)-sp.cos(K2)-sp.cos(K3) + 3)**order
        # Constant band splitting
        if zeeman:
            hx -= self.m_zee_x
            hy -= self.m_zee_y
            hz -= self.m_zee_z

        super().__init__(ho, hx, hy, hz)


class BiTeResummed(TwoBandSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 c2=sp.Symbol('c2', real=True),
                 A=sp.Symbol('A', real=True),
                 r=sp.Symbol('r', real=True),
                 ksym=sp.Symbol('ksym', real=True),
                 kasym=sp.Symbol('kasym', real=True),
                 zeeman=False):

        k = sp.sqrt(self.kx**2 + self.ky**2)
        C2 = (c2/ksym**2)/(1+(k/ksym)**2)
        R = (r/kasym**2)/(1+(k/kasym)**4)
        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*self.kx*(self.kx**2 - 3*self.ky**2)

        if zeeman:
            hx -= self.m_zee_x
            hy -= self.m_zee_y
            hz -= self.m_zee_z

        super().__init__(ho, hx, hy, hz)


class Graphene(TwoBandSystem):
    """
    Graphene model
    """

    def __init__(self, t=sp.Symbol('t')):
        a1 = self.kx
        a2 = -1/2 * self.kx + sp.sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sp.sqrt(3)/2 * self.ky

        ho = 0
        hx = t*(sp.cos(a1)+sp.cos(a2)+sp.cos(a3))
        hy = t*(sp.sin(a1)+sp.sin(a2)+sp.sin(a3))
        hz = 0

        super().__init__(ho, hx, hy, hz)


class QWZ(TwoBandSystem):
    """
    Qi-Wu-Zhang model of a 2D Chern insulator
    """

    def __init__(self, t=sp.Symbol('t'), m=sp.Symbol('m'), order=sp.oo):
        n = order+1

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

        super().__init__(t*ho, t*hx, t*hy, t*hz)


class Dirac(TwoBandSystem):
    """
    Generic Dirac cone Hamiltonian
    """

    def __init__(self, vx=sp.Symbol('vx'), vy=sp.Symbol('vy'),
                 m=sp.Symbol('m')):

        ho = 0
        hx = vx*self.kx
        hy = vy*self.ky
        hz = m

        super().__init__(ho, hx, hy, hz)


class Test(TwoBandSystem):
    def __init__(self, A=sp.Symbol('A', real=True),
                 a=sp.Symbol('a', real=True),
                 mx=0, mz=0, zeeman=False):

        ho = 0
        hx = mx
        hy = 0
        hz = A*(2 + mz - sp.cos((2*a/3)*self.kx) - sp.cos((2*a/3)*self.ky))

        if (zeeman):
            hx -= self.m_zee_x
            hy -= self.m_zee_y
            hz -= self.m_zee_z

        super().__init__(ho, hx, hy, hz)


class Parabolic(TwoBandSystem):
    def __init__(self, A=sp.Symbol('A', real=True),
                 mz=0, zeeman=False):

        ho = 0
        hx = A*(self.ky**2)
        hy = A*(self.kx**2)
        hz = mz

        if (zeeman):
            hx -= self.m_zee_x
            hy -= self.m_zee_y
            hz -= self.m_zee_z

        super().__init__(ho, hx, hy, hz)


class Semiconductor(TwoBandSystem):
    """
    Generic Semiconductor Hamiltonian
    """

    def __init__(self, A=sp.Symbol('A'), mx=sp.Symbol('mx'),
                 mz=sp.Symbol('mz'), a=sp.Symbol('a'), align=False):
        ho = 0
        hx = mx
        hy = 0

        if (align):
            hz = A*(2 + mz - sp.cos((2*a/3)*self.kx) - sp.cos((2*a/3)*self.ky))
        else:
            hz = (A/4)*(2 - sp.cos((2*a/3)*self.kx) - sp.cos((2*a/3)*self.ky))

        super().__init__(ho, hx, hy, hz)
