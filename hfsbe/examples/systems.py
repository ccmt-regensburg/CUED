from sympy import *


class TwoBandSystems():
    def __init__(self):
        """
        Returns the symbolic Hamiltonian and wave function of the system with 
        the given name.
        
        """
        
        self.so = Matrix([[1, 0], [0, 1]])
        self.sx = Matrix([[0, 1], [1, 0]])
        self.sy = Matrix([[0, -I], [I, 0]])
        self.sz = Matrix([[1, 0], [0, -1]])

        self.kx = Symbol("kx")
        self.ky = Symbol("ky")
    
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
        
        esoc = sqrt(hx**2 + hy**2 + hz**2)
        e = [ho - esoc, ho + esoc]
                
        wfv = Matrix([-hx + I*hy, hz + esoc])
        wfc = Matrix([hz + esoc, hx + I*hy])
        wfv_h = Matrix([-hx - I*hy, hz + esoc])
        wfc_h = Matrix([hz + esoc, hx - I*hy])
        norm = sqrt(2*(esoc + hz)*esoc)
        
        wf = [wfv/norm, wfv_h/norm, wfc/norm, wfc_h/norm]

        h = ho*self.so + hx*self.sx + hy*self.sy + hz*self.sz
        
        return h, e, wf

    
    def haldane(self):
        """
        Haldane model
        """
        t1 = Symbol("t1")
        t2 = Symbol("t2")
        m = Symbol('m')
        phi = Symbol("phi")

        a1 = self.kx
        a2 = -1/2 * self.kx + sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sqrt(3)/2 * self.ky

        b1 = sqrt(3) * self.ky
        b2 = -3/2 * self.kx - sqrt(3)/2 * self.ky
        b3 = 3/2 * self.kx - sqrt(3)/2 * self.ky

        ho = 2*t2*cos(phi)*(cos(b1)+cos(b2)+cos(b3))
        hx = t1*(cos(a1)+cos(a2)+cos(a3))
        hy = t1*(sin(a1)+sin(a2)+sin(a3))
        hz = m - 2*t2*sin(phi)*(sin(b1)+sin(b2)+sin(b3))

        return self.__eigensystem(ho, hx, hy, hz)

    
    def bite(self):
        """
        Bismuth Telluride topological insulator model
        """
        C0 = Symbol("C0")
        C2 = Symbol("C2")
        A = Symbol("A")
        R = Symbol("R")

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2)
        
        return self.__eigensystem(ho, hx, hy, hz)

    
    def graphene(self):
        """
        Graphene model
        """
        t = Symbol("t")
        
        a1 = self.kx
        a2 = -1/2 * self.kx + sqrt(3)/2 * self.ky
        a3 = -1/2 * self.kx - sqrt(3)/2 * self.ky

        ho = 0
        hx = t*(cos(a1)+cos(a2)+cos(a3))
        hy = t*(sin(a1)+sin(a2)+sin(a3))
        hz = 0

        return self.__eigensystem(ho, hx, hy, hz) 

    
    def qwz(self):
        """
        Qi-Wu-Zhang model of a 2D Chern insulator
        """
        m = Symbol("m")

        ho = 0
        hx = sin(self.kx)
        hy = sin(self.ky)
        hz = m - cos(self.kx) - cos(self.ky)
        
        return self.__eigensystem(ho, hx, hy, hz)
