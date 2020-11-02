import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from sbe.utility import matrix_to_njit_functions, to_numpy_function

plt.rcParams['figure.figsize'] = [12, 15]
plt.rcParams['text.usetex'] = True


class SymbolicCurvature():
    """
    This class constructs the Berry curvature from a given symbolic
    Berry connection calculated by SymbolicDipole.
    Also the Hamiltonian is needed to recover the original set of symbols
    """

    def __init__(self, h, Ax, Ay):
        """
        Parameters
        ----------
        h : Matrix of Symbol
            Original system Hamiltonian
        Ax, Ay : Matrix of Symbol
            x, y components of all dipole fields
        """

        self.kx = sp.Symbol('kx', real=True)
        self.ky = sp.Symbol('ky', real=True)

        self.B = sp.diff(Ax, self.ky) - sp.diff(Ay, self.kx)

        hsymbols = h.free_symbols
        self.Bfjit = matrix_to_njit_functions(self.B, hsymbols)

        self.Bf = to_numpy_function(self.B)

        self.B_eval = None

    def evaluate(self, kx, ky, **fkwargs):
        # Evaluate all kpoints without BZ

        self.B_eval = self.Bf(kx=kx, ky=ky, **fkwargs)

        return self.B_eval


    def plot_curvcature_3d(self, kx, ky, title="Curvature field"):
        print("Not Implemented")


    def plot_curvature_contour(self, kx, ky, vidx=0, cidx=1,
                               title=None, vname=None, cname=None,
                               xlabel=None, ylabel=None, clabel=None):
        """
        Plot the specified Berry curvature scalar field.

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
            title = "Berry curvature"
        if (vname is None):
            vname = vidx
        if (cname is None):
            cname = cidx
        if (xlabel is None):
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if (ylabel is None):
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if (clabel is None):
            clabel = r'Berry curvature $[\mathrm{a.u.}]$'

        Be = self.B_eval

        if (Be is None):
            raise RuntimeError("Error: The curvature fields first need to"
                               " be evaluated on a kgrid to plot them. "
                               " Call evaluate before plotting.")

        Be = np.real(Be)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title, fontsize=16)

        valence = ax[0].scatter(kx, ky, s=2, c=Be[vidx, vidx], cmap="cool")
        ax[0].set_title(r"$\Omega_{" + str(vname) + str(vname) + "}$")
        ax[0].axis('equal')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0], label=clabel)

        conduct = ax[1].scatter(kx, ky, s=2, c=Be[cidx, cidx], cmap="cool")
        ax[1].set_title(r"$\Omega_{" + str(cname) + str(cname) + "}$")
        ax[1].axis('equal')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[1], label=clabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
