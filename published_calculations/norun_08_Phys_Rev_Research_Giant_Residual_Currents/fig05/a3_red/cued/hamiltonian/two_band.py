import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from cued.utility import list_to_njit_functions, matrix_to_njit_functions
from cued.utility import evaluate_njit_matrix
plt.rcParams['figure.figsize'] = [12, 15]
plt.rcParams['text.usetex'] = True


class TwoBandHamiltonianSystem():
    so = sp.Matrix([[1, 0], [0, 1]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    m_zee_x = sp.Symbol('m_zee_x', real=True)
    m_zee_y = sp.Symbol('m_zee_y', real=True)
    m_zee_z = sp.Symbol('m_zee_z', real=True)

    def __init__(self, ho, hx, hy, hz, offdiagonal_k=False, test=False, kdotp=None):
        """
        Generates the symbolic Hamiltonian, wave functions and
        energies.

        Parameters
        ----------
        ho, hx, hy, hz : Symbol
            Wheter to additionally return energy and wave function derivatives
        """
        self.system = 'ana'

        self.n = 2

        self.ho = ho
        self.hx = hx
        self.hy = hy
        self.hz = hz

        self.offdiag_k = offdiagonal_k
        self.test = test
        self.kdotp = kdotp

        self.h = self.__hamiltonian()
        self.hsymbols = self.h.free_symbols
        self.hderiv = self.__hamiltonian_derivatives()

        self.degenerate_eigenvalues = False

        self.e = None
        self.ederiv = None

        self.efjit = None
        self.ederivjit = None

        # Get set when eigensystem is called (gauge needed)
        self.U = None             # Normalised eigenstates
        self.U_h = None           # Hermitian conjugate
        self.U_no_norm = None     # Unnormalised eigenstates
        self.U_h_no_norm = None   # Hermitian conjugate

        self.Ujit = None
        self.Ujit_h = None

        # Evaluated fields
        self.Ax_eval = None
        self.Ay_eval = None

        # Offdiagonal k elements
        self.Ax_offk = None
        self.Ay_offk = None

        self.Axfjit_offk = None
        self.Ayfjit_offk = None

        self.B_eval = None

        # Get set when evaluate_energy is called
        self.e_eval = None
        self.ederiv_eval = None

        self.e_in_path = None   #set when eigensystem_dipole_path is called
        self.wf_in_path = None

        self.dipole_path_x = None
        self.dipole_path_y = None

        self.dipole_in_path = None
        self.dipole_ortho = None

        self.dipole_derivative = None
        self.dipole_derivative_jit = None
        self.dipole_derivative_in_path = None

    def __hamiltonian(self):
        return self.ho*self.so + self.hx*self.sx + self.hy*self.sy \
            + self.hz*self.sz

    def __hamiltonian_derivatives(self):
        return [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]


    def __energies(self):
        esoc = sp.sqrt(self.hx**2 + self.hy**2 + self.hz**2)
        return [self.ho - esoc, self.ho + esoc]

    def __energy_derivatives(self):
        """
        Calculate the derivative of the energy bands. Order is
        de[0]/dkx, de[0]/dky, de[1]/dkx, de[1]/dky
        """
        ed = []
        for e in self.e:
            ed.append(sp.diff(e, self.kx))
            ed.append(sp.diff(e, self.ky))
        return ed

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

    def __fields(self, U, U_h):
        dUx = sp.diff(U, self.kx)
        dUy = sp.diff(U, self.ky)
        # Minus sign is the charge
        return -sp.I*U_h * dUx, -sp.I*U_h * dUy

    def __kdotp_fields(self, kdotp, ev, ec):
        Ax = sp.Matrix([[0, kdotp[0]/(ec-ev)], [np.conjugate(kdotp[0])/(ec-ev), 0]])
        Ay = sp.Matrix([[0, kdotp[1]/(ec-ev)], [np.conjugate(kdotp[1])/(ec-ev), 0]])
        return Ax, Ay

    def eigensystem_dipole_path(self, path, P):

        # Set eigenfunction first time eigensystem_dipole_path is called
        if self.e is None:
            self.make_eigensystem_dipole(P)

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        pathlen = path[:,0].size
        self.e_in_path = np.zeros([pathlen, P.n], dtype=P.type_real_np)

        if P.dm_dynamics_method == 'semiclassics':
            self.dipole_path_x = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)
            self.dipole_path_y = np.zeros([pathlen, P.n, P.n], dtype=P.type_complex_np)
            self.Ax_path = evaluate_njit_matrix(self.Axfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.Ay_path = evaluate_njit_matrix(self.Ayfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.Bcurv = evaluate_njit_matrix(self.Bfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        else:
            # Calculate the dipole components along the path
            self.dipole_path_x = evaluate_njit_matrix(self.Axfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.dipole_path_y = evaluate_njit_matrix(self.Ayfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        for n, e in enumerate(self.efjit):
            self.e_in_path[:, n] = e(kx=kx_in_path, ky=ky_in_path)

        self.wf_in_path = evaluate_njit_matrix(self.Ujit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        self.dipole_in_path = P.E_dir[0]*self.dipole_path_x + P.E_dir[1]*self.dipole_path_y
        self.dipole_ortho = P.E_ort[0]*self.dipole_path_x + P.E_ort[1]*self.dipole_path_y

        if P.dm_dynamics_method == 'EEA':
            self.dipole_derivative_in_path = evaluate_njit_matrix(self.dipole_derivative_jit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)


    def make_eigensystem_dipole(self, P):

        self.e = self.__energies()
        self.ederiv = self.__energy_derivatives()

                # Jitted Hamiltonian and energies
        self.hfjit = matrix_to_njit_functions(self.h, self.hsymbols, dtype=P.type_complex_np)
        self.hderivfjit = [matrix_to_njit_functions(hd, self.hsymbols, dtype=P.type_complex_np)
                           for hd in self.hderiv]

        self.efjit = list_to_njit_functions(self.e, self.hsymbols, dtype=P.type_complex_np)
        self.ederivfjit = list_to_njit_functions(self.ederiv, self.hsymbols, dtype=P.type_complex_np)

        self.eigensystem(P)

        if self.kdotp is None:
            self.Ax, self.Ay = self.__fields(self.U, self.U_h)
        else:
            self.Ax, self.Ay = self.__kdotp_fields(self.kdotp, self.e[0], self.e[1])

        # Njit function and function arguments
        self.Axfjit = matrix_to_njit_functions(self.Ax, self.hsymbols, dtype=P.type_complex_np)
        self.Ayfjit = matrix_to_njit_functions(self.Ay, self.hsymbols, dtype=P.type_complex_np)

        if self.offdiag_k:
            self.offdiagonal_k(self.U)

        # Curvature

        self.B = sp.diff(self.Ax, self.ky) - sp.diff(self.Ay, self.kx)
        self.Bfjit = matrix_to_njit_functions(self.B, self.hsymbols, dtype=P.type_complex_np)

        if P.dm_dynamics_method == 'EEA':
            self.dipole_derivative = P.E_dir[0] * P.E_dir[0] * sp.diff(self.Ax, self.kx) \
                + P.E_dir[0] * P.E_dir[1] * ( sp.diff(self.Ax, self.ky) + sp.diff(self.Ay, self.kx) ) \
                + P.E_dir[1] * P.E_dir[1] * sp.diff(self.Ay, self.ky)
            self.dipole_derivative_jit = matrix_to_njit_functions(self.dipole_derivative, self.hsymbols, dtype=P.type_complex_np)

    def eigensystem(self, P):
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
        gidx = P.gidx
        self.U, self.U_h, self.U_no_norm, self.U_h_no_norm = \
            self.__wave_function(gidx=gidx)

        self.Ujit = matrix_to_njit_functions(self.U, self.hsymbols, P.type_complex_np)
        self.Ujit_h = matrix_to_njit_functions(self.U_h, self.hsymbols, P.type_complex_np)

    def evaluate_energy(self, kx, ky, **fkwargs):
        # Evaluate all kpoints without BZ
        self.e_eval = []

        for ef in self.efjit:
            self.e_eval.append(ef(kx=kx, ky=ky, **fkwargs))
        return self.e_eval

    def evaluate_ederivative(self, kx, ky, **fkwargs):
        self.ederiv_eval = []
        # Evaluate all kpoints without BZ
        for ederivf in self.ederivfjit:
            self.ederiv_eval.append(ederivf(kx=kx, ky=ky, **fkwargs))
        return self.ederiv_eval

    def evaluate_dipole(self, kx, ky, **fkwargs):
        """
        Transforms the symbolic expression for the
        berry connection/dipole moment matrix to an expression
        that is numerically evaluated.

        Parameters
        ----------
        kx, ky : np.ndarray
            array of all point combinations
        fkwargs :
            keyword arguments passed to the symbolic expression
        """
        # Evaluate all kpoints without BZ
        self.Ax_eval = evaluate_njit_matrix(self.Axfjit, kx, ky, **fkwargs)
        self.Ay_eval = evaluate_njit_matrix(self.Ayfjit, kx, ky, **fkwargs)
        return self.Ax_eval, self.Ay_eval

    def offdiagonal_k(self, wf):
        kxp = sp.Symbol('kxp', real=True)
        kyp = sp.Symbol('kyp', real=True)

        Ukp_h = wf[1].subs(self.kx, kxp).subs(self.ky, kyp)
        self.Ax_offk, self.Ay_offk = self.__fields(wf[0], Ukp_h)

        self.Axfjit_offk = matrix_to_njit_functions(self.Ax_offk,
                                                    self.hsymbols, kpflag=True)
        self.Ayfjit_offk = matrix_to_njit_functions(self.Ay_offk,
                                                    self.hsymbols, kpflag=True)

    def evaluate_curvature(self, kx, ky, **fkwargs):
        # Evaluate all kpoints without BZ

        self.B_eval = evaluate_njit_matrix(self.Bfjit, kx, ky, **fkwargs)

        return self.B_eval

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


    def plot_dipoles(self, kx, ky, vidx=0, cidx=1,
                     title=None, vname=None, cname=None,
                     xlabel=None, ylabel=None, clabel=None, nolog=False,
                     savename=None):
        """
        Plot two dipole fields corresponding to the indices vidx and
        cidx

        Parameters
        ----------
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
        nolog: bool
            If the color coding should be logarithmic or linear
        """
        if vname is None:
            vname = vidx
        if cname is None:
            cname = cidx
        if xlabel is None:
            xlabel = r'$k_x [\mathrm{at.u.}]$'
        if ylabel is None:
            ylabel = r'$k_y [\mathrm{at.u.}]$'
        if clabel is None:
            if nolog:
                clabel = r'$[\mathrm{at.u.}]$'
            else:
                clabel = r'$[\mathrm{at.u.}]$ in $\log_{10}$ scale'

        Axe, Aye = self.Ax_eval, self.Ay_eval

        if Axe is None or Aye is None:
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
        if title is not None:
            fig.suptitle(title, fontsize=16)

        if nolog:
            valence_c = norm_r[:, vidx, vidx]
            conduct_c = norm_r[:, cidx, cidx]
            condval_c_r = norm_r[:, cidx, vidx]
            condval_c_i = norm_i[:, cidx, vidx]
        else:
            valence_c = np.log10(norm_r[:, vidx, vidx])
            conduct_c = np.log10(norm_r[:, cidx, cidx])
            condval_c_r = np.log10(norm_r[:, cidx, vidx])
            condval_c_i = np.log10(norm_i[:, cidx, vidx])

        valence = ax[0, 0].quiver(kx, ky,
                                  Axe_rn[:, vidx, vidx], Aye_rn[:, vidx, vidx],
                                  valence_c, angles='xy', cmap='cool', width=0.007)
        current_name = r"$\mathrm{Re}(\mathbf{d}_{" + str(vname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[0, 0].set_title(current_name)
        ax[0, 0].axis('equal')
        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0, 0], label=current_abs_name + clabel)

        conduct = ax[0, 1].quiver(kx, ky,
                                  Axe_rn[:, cidx, cidx], Aye_rn[:, cidx, cidx],
                                  conduct_c, angles='xy', cmap='cool', width=0.007)
        current_name = r"$\mathrm{Re}(\mathbf{d}_{" + str(cname) + str(cname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[0, 1].set_title(current_name)
        ax[0, 1].axis('equal')
        ax[0, 1].set_xlabel(xlabel)
        ax[0, 1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[0, 1], label=current_abs_name + clabel)

        dipreal = ax[1, 0].quiver(kx, ky,
                                  Axe_rn[:, cidx, vidx], Aye_rn[:, cidx, vidx],
                                  condval_c_r, angles='xy', cmap='cool', width=0.007)
        current_name = r"$\mathrm{Re}(\mathbf{d}_{" + str(cname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[1, 0].set_title(current_name)
        ax[1, 0].axis('equal')
        ax[1, 0].set_xlabel(xlabel)
        ax[1, 0].set_ylabel(ylabel)
        plt.colorbar(dipreal, ax=ax[1, 0], label=current_abs_name + clabel)

        dipimag = ax[1, 1].quiver(kx, ky,
                                  Axe_in[:, cidx, vidx], Aye_in[:, cidx, vidx],
                                  condval_c_i, angles='xy', cmap='cool', width=0.007)
        current_name = r"$\mathrm{Im}(\mathbf{d}_{" + str(cname) + str(vname) + "})$"
        current_abs_name = r'$|$' + current_name + r'$|$'
        ax[1, 1].set_title(current_name)
        ax[1, 1].axis('equal')
        ax[1, 1].set_xlabel(xlabel)
        ax[1, 1].set_ylabel(ylabel)
        plt.colorbar(dipimag, ax=ax[1, 1], label=current_abs_name + clabel)

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()


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
        if title is None:
            title = "Berry curvature"
        if vname is None:
            vname = vidx
        if cname is None:
            cname = cidx
        if xlabel is None:
            xlabel = r'$k_x [\mathrm{a.u.}]$'
        if ylabel is None:
            ylabel = r'$k_y [\mathrm{a.u.}]$'
        if clabel is None:
            clabel = r'Berry curvature $[\mathrm{a.u.}]$'

        Be = self.B_eval

        if Be is None:
            raise RuntimeError("Error: The curvature fields first need to"
                               " be evaluated on a kgrid to plot them. "
                               " Call evaluate before plotting.")

        Be = np.real(Be)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title, fontsize=16)

        valence = ax[0].scatter(kx, ky, s=2, c=Be[:, vidx, vidx], cmap="cool")
        ax[0].set_title(r"$\Omega_{" + str(vname) + str(vname) + "}$")
        ax[0].axis('equal')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        plt.colorbar(valence, ax=ax[0], label=clabel)

        conduct = ax[1].scatter(kx, ky, s=2, c=Be[:, cidx, cidx], cmap="cool")
        ax[1].set_title(r"$\Omega_{" + str(cname) + str(cname) + "}$")
        ax[1].axis('equal')
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        plt.colorbar(conduct, ax=ax[1], label=clabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
