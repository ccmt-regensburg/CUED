import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

import hfsbe.check.symbolic_checks as symbolic_checks
from hfsbe.brillouin import evaluate_matrix_field as evaldip
from hfsbe.utility import matrix_to_njit_functions, \
    to_njit_function, to_numpy_function

plt.rcParams['figure.figsize'] = [12, 15]
plt.rcParams['text.usetex'] = True


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
n        wf : np.ndarray of Symbol
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

        self.kx = sp.Symbol('kx', real=True)
        self.ky = sp.Symbol('ky', real=True)

        self.h = h
        self.e = e
        self.U = wf[0]
        self.U_h = wf[1]

        self.Ax, self.Ay = self.__fields()

        # Numpy function and function arguments
        self.Axfjit = matrix_to_njit_functions(self.Ax)
        self.Ayfjit = matrix_to_njit_functions(self.Ay)

        self.Axf = to_numpy_function(self.Ax)
        self.Ayf = to_numpy_function(self.Ay)

        # Evaluated fields
        self.Ax_eval = None
        self.Ay_eval = None

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
            self.Ax_eval = self.Axf(kx=kx, ky=ky, **fkwargs)
            self.Ay_eval = self.Ayf(kx=kx, ky=ky, **fkwargs)
            return self.Ax_eval, self.Ay_eval

        # Add a BZ and throw error if kx, ky is outside
        self.Ax_eval = evaldip(self.Axf, kx, ky, self.b1, self.b2,
                               hamr=hamr, eps=eps,
                               **fkwargs)
        self.Ay_eval = evaldip(self.Ayf, kx, ky, self.b1, self.b2,
                               hamr=hamr, eps=eps,
                               **fkwargs)

        return self.Ax_eval, self.Ay_eval

    def plot_dipoles(self, kx, ky, vidx=0, cidx=1,
                     title=None, vname=None, cname=None,
                     xlabel=None, ylabel=None, clabel=None):
        """
        Plot two dipole fields corresponding to the indices vidx and
        cidx

        Parameters:
        kx, ky : np.ndarray
            array of all point combinations (same as evaluate)
        vidx, cidx : int
            Index of the first and second band to evaluate
        title: string
            Title of the plot
        vname, cname:
            Index names of the valence and conduction band
        xlabel, ylabel, clabel : string
            Label of x, y- axis and colorbar
        """
        if (title is None):
            title = "Dipole fields"
        if (vname is None):
            vname = vidx
        if (cname is None):
            cname = cidx
        if (xlabel is None):
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if (ylabel is None):
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if (clabel is None):
            clabel = r'$[\mathrm{a.u.}]$ in $\log_{10}$ scale'

        Axe, Aye = self.Ax_eval, self.Ay_eval

        if (Axe is None or Aye is None):
            raise RuntimeError("Error: The dipole fields first need to"
                               " be evaluated on a kgrid to plot them. "
                               " Call evaluate before plotting.")

        Axe_r, Axe_i = np.real(Axe), np.imag(Axe)
        Aye_r, Aye_i = np.real(Aye), np.imag(Aye)

        norm_r = np.sqrt(Axe_r**2 + Aye_r**2)
        norm_i = np.sqrt(Axe_i**2 + Aye_i**2)

        Axe_rn, Axe_in = Axe_r/norm_r, Axe_i/norm_i
        Aye_rn, Aye_in = Aye_r/norm_r, Aye_i/norm_i

        fig, ax = plt.subplots(2, 2)
        fig.suptitle(title, fontsize=16)

        valence = ax[0, 0].quiver(kx, ky,
                                  Axe_rn[vidx, vidx], Aye_rn[vidx, vidx],
                                  np.log10(norm_r[vidx, vidx]),
                                  angles='xy', cmap='cool')
        current_name = r"$\Re(\vec{A}_{" + str(vname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[0, 0].set_title(current_name)
        ax[0, 0].axis('equal')
        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0, 0], label=current_abs_name + clabel)

        conduct = ax[0, 1].quiver(kx, ky,
                                  Axe_rn[cidx, cidx], Aye_rn[cidx, cidx],
                                  np.log10(norm_r[cidx, cidx]),
                                  angles='xy', cmap='cool')
        current_name = r"$\Re(\vec{A}_{" + str(cname) + str(cname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[0, 1].set_title(current_name)
        ax[0, 1].axis('equal')
        ax[0, 1].set_xlabel(xlabel)
        ax[0, 1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[0, 1], label=current_abs_name + clabel)

        dipreal = ax[1, 0].quiver(kx, ky,
                                  Axe_rn[cidx, vidx], Aye_rn[cidx, vidx],
                                  np.log10(norm_r[cidx, vidx]),
                                  angles='xy', cmap='cool')
        current_name = r"$\Re(\vec{A}_{" + str(cname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[1, 0].set_title(current_name)
        ax[1, 0].axis('equal')
        ax[1, 0].set_xlabel(xlabel)
        ax[1, 0].set_ylabel(ylabel)
        plt.colorbar(dipreal, ax=ax[1, 0], label=current_abs_name + clabel)

        dipimag = ax[1, 1].quiver(kx, ky,
                                  Axe_in[cidx, vidx], Aye_in[cidx, vidx],
                                  np.log10(norm_i[cidx, vidx]),
                                  angles='xy', cmap='cool')
        current_name = r"$\Im(\vec{A}_{" + str(cname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[1, 1].set_title(current_name)
        ax[1, 1].axis('equal')
        ax[1, 1].set_xlabel(xlabel)
        ax[1, 1].set_ylabel(ylabel)
        plt.colorbar(dipimag, ax=ax[1, 1], label=current_abs_name + clabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
