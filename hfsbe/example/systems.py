import sympy as sp


class TwoBandSystems():
    def __init__(self, e_deriv=False):
        """
        Returns the symbolic Hamiltonian and wave function of the system with
        the given name.

        Parameters
        ----------
        e_deriv : bool
            Wheter to additionally return energy and wave function derivatives
        """
        self.e_deriv = e_deriv
        self.so = sp.Matrix([[1, 0], [0, 1]])
        self.sx = sp.Matrix([[0, 1], [1, 0]])
        self.sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
        self.sz = sp.Matrix([[1, 0], [0, -1]])

        self.kx = sp.Symbol('kx')
        self.ky = sp.Symbol('ky')

    def __eigensystem(self, ho, hx, hy, hz):
        """
        Generic form of Hamiltonian, energies and wave functions in a two band
        Hamiltonian.

        Parameters
        ----------
        ho, hx, hy, hz : Symbol
            Hamiltonian part proportional to unit, sx, sy, sz Pauli matrix

        Returns
        -------
        h : Symbol
            Hamiltonian of the system
        e : list of Symbol
            Valence and conduction band energies; in this order
        wf : list of Symbol
            Valence and conduction band wave function; in this order
        """
        esoc = sp.sqrt(hx**2 + hy**2 + hz**2)
        e = [ho - esoc, ho + esoc]

        wfv = sp.Matrix([-hx + sp.I*hy, hz + esoc])
        wfc = sp.Matrix([hz + esoc, hx + sp.I*hy])
        wfv_h = sp.Matrix([-hx - sp.I*hy, hz + esoc])
        wfc_h = sp.Matrix([hz + esoc, hx - sp.I*hy])
        norm = sp.sqrt(2*(esoc + hz)*esoc)

        U = (wfv/norm).row_join(wfc/norm)
        U_h = (wfv_h/norm).T.col_join((wfc_h/norm).T)

        h = ho*self.so + hx*self.sx + hy*self.sy + hz*self.sz

        if self.e_deriv:
            return h, e, [U, U_h], self.__e_deriv(e)
        else:
            return h, e, [U, U_h]

    def __e_deriv(self, energies):
        """
        Calculate the derivative of the energy bands. Order is
        de[0]/dkx, de[0]/dky, de[1]/dkx, de[1]/dky
        """
        ed = []
        for e in energies:
            ed.append(sp.diff(e, self.kx))
            ed.append(sp.diff(e, self.ky))
        return ed

    def haldane(self):
        """
        Haldane model
        """
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

        return self.__eigensystem(ho, hx, hy, hz)

    def bite(self):
        """
        Bismuth Telluride topological insulator model
        """
        C0 = sp.Symbol('C0')
        C2 = sp.Symbol('C2')
        A = sp.Symbol('A')
        R = sp.Symbol('R')

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2)

        return self.__eigensystem(ho, hx, hy, hz)

    def graphene(self):
        """
        Graphene model
        """
        t = sp.Symbol('t')

        a1 = self.kx
        a2 = -1/2 * self.kx + sp.sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sp.sqrt(3)/2 * self.ky

        ho = 0
        hx = t*(sp.cos(a1)+sp.cos(a2)+sp.cos(a3))
        hy = t*(sp.sin(a1)+sp.sin(a2)+sp.sin(a3))
        hz = 0

        return self.__eigensystem(ho, hx, hy, hz)

    def qwz(self):
        """
        Qi-Wu-Zhang model of a 2D Chern insulator
        """
        m = sp.Symbol('m')

        ho = 0
        hx = sp.sin(self.kx)
        hy = sp.sin(self.ky)
        hz = m - sp.cos(self.kx) - sp.cos(self.ky)

        return self.__eigensystem(ho, hx, hy, hz)

    def dirac(self):
        """
        Generic Dirac cone Hamiltonian
        """
        m = sp.Symbol('m')

        ho = 0
        hx = self.kx
        hy = self.ky
        hz = m

        return self.__eigensystem(ho, hx, hy, hz)
