import numpy as np
import sympy as sp

from cued.hamiltonian import TwoBandHamiltonianSystem, NBandHamiltonianSystem, NBandBandstructureDipoleSystem


############################################################################################
# Two Band Hamiltonians for analytic evaluation
############################################################################################

class Haldane(TwoBandHamiltonianSystem):
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


class two_site_semiconductor(TwoBandHamiltonianSystem):
    """
    k-dependent Hamiltonian of 1d tight-binding chain with two sites in the unit cell
    !!! ONLY WORKS TOGETHER WITH gidx = None, REQUIRES 1D BRILLOUIN ZONE OF LENGTH 2*pi/lattice_const !!!
    """

    def __init__(self,
                 lattice_const=sp.Symbol('lattice_const', real=True),
                 hopping=sp.Symbol('hopping', real=True),
                 onsite_energy_difference=sp.Symbol('onsite_energy_difference', real=True)) :

        ho = 1.0E-30*self.ky
        hx = hopping * (1 + sp.cos (self.kx * lattice_const))
        hy = 0
        hz = onsite_energy_difference/2

        super().__init__(ho, hx, hy, hz)


class one_site_semiconductor(TwoBandHamiltonianSystem):
    """
    k-dependent Hamiltonian of 1d tight-binding chain with two sites in the unit cell
    !!! ONLY WORKS TOGETHER WITH gidx = None, REQUIRES 1D BRILLOUIN ZONE OF LENGTH 2*pi/lattice_const !!!
    """

    def __init__(self,
                 lattice_const=sp.Symbol('lattice_const', real=True),
                 hopping=sp.Symbol('hopping', real=True)):

        ho = hopping * (1 - 2*sp.cos (self.kx * lattice_const))
        hx = 1.0e-50*self.ky
        hy = 1.0e-50
        hz = 1.0e-50

        super().__init__(ho, hx, hy, hz)



class BiTe(TwoBandHamiltonianSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0):

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2) + mz

        if (not np.isclose(kcut, 0)):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor

        super().__init__(ho, hx, hy, hz)


class BiTePeriodic(TwoBandHamiltonianSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, A=sp.Symbol('A', real=True),
                 C2=sp.Symbol('C2', real=True),
                 R=sp.Symbol('R', real=True),
                 a=sp.Symbol('a', real=True),
                 mw=1, order=4):

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

        super().__init__(ho, hx, hy, hz)


class BiTeResummed(TwoBandHamiltonianSystem):
    """
    Bismuth Telluride topological insulator model
    """

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 c2=sp.Symbol('c2', real=True),
                 A=sp.Symbol('A', real=True),
                 r=sp.Symbol('r', real=True),
                 ksym=sp.Symbol('ksym', real=True),
                 kasym=sp.Symbol('kasym', real=True)):

        k = sp.sqrt(self.kx**2 + self.ky**2)
        C2 = (c2/ksym**2)/(1+(k/ksym)**2)
        R = (r/kasym**2)/(1+(k/kasym)**4)
        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*self.kx*(self.kx**2 - 3*self.ky**2)

        super().__init__(ho, hx, hy, hz)


class Graphene(TwoBandHamiltonianSystem):
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


class QWZ(TwoBandHamiltonianSystem):
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


class Dirac(TwoBandHamiltonianSystem):
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


class Test(TwoBandHamiltonianSystem):
    def __init__(self, A=sp.Symbol('A', real=True),
                 a=sp.Symbol('a', real=True),
                 mx=0, mz=0):

        ho = 0
        hx = mx
        hy = 0
        hz = A*(2 + mz - sp.cos((2*a/3)*self.kx) - sp.cos((2*a/3)*self.ky))

        super().__init__(ho, hx, hy, hz)


class Parabolic(TwoBandHamiltonianSystem):
    def __init__(self, A=sp.Symbol('A', real=True),
                 mz=0):

        ho = 0
        hx = A*(self.ky**2)
        hy = A*(self.kx**2)
        hz = mz

        super().__init__(ho, hx, hy, hz)


class Semiconductor(TwoBandHamiltonianSystem):
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


############################################################################################
# N Band Hamiltonians for numeric evaluation
############################################################################################

class BiTe_num(NBandHamiltonianSystem):
    '''
       BiTe Hamiltonian for numerical evaluations
    '''
    
    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0):

        so = sp.Matrix([[1, 0], [0, 1]])
        sx = sp.Matrix([[0, 1], [1, 0]])
        sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
        sz = sp.Matrix([[1, 0], [0, -1]])

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2) + mz

        if (not np.isclose(kcut, 0)):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor
        h = ho*so + hx*sx + hy*sy + hz*sz

        super().__init__(h)

class BiTe_num_3_bands(NBandHamiltonianSystem):
    '''
        Artificial 3Band model with Dirac cone for first two bands, zero else
    '''
    
    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0):

        so = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        sy = sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]])
        sz = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        ho = C0 + C2*(self.kx**2 + self.ky**2)
        hx = A*self.ky
        hy = -A*self.kx
        hz = 2*R*(self.kx**3 - 3*self.kx*self.ky**2) + mz

        if (not np.isclose(kcut, 0)):
            ratio = (self.kx**2 + self.ky**2)/kcut**2
            cutfactor = 1/(1+(ratio))
            hz *= cutfactor
        h = ho*so + hx*sx + hy*sy + hz*sz

        super().__init__(h)


############################################################################################
# Bandstructures and Dipoles for kp-evaluation
############################################################################################

class ExampleTwoBand(NBandBandstructureDipoleSystem):

    def __init__(self, 
                a = sp.Symbol('a', real=True), 
                prefac_x = sp.Symbol('prefac_x'), 
                prefac_y = sp.Symbol('prefac_y') , flag = None):


        ev = sp.cos(a*self.kx + self.ky)
        ec = sp.sin(a*self.ky)

        e = [ev, ec]
        n = 2 

        super().__init__(e, prefac_x, prefac_y, n, flag)

class BiTeBandstructure(NBandBandstructureDipoleSystem):

    def __init__(self, vF = sp.Symbol('vF', real=True),
                prefac_x = sp.Symbol('prefac_x'),
                prefac_y = sp.Symbol('prefac_y'), flag = None):
                
        ev = - vF * sp.sqrt( self.kx**2 + self.ky**2 )
        ec =   vF * sp.sqrt( self.kx**2 + self.ky**2 )

        e = [ev, ec]
        n = 2

        super().__init__(e, prefac_x, prefac_y, n, flag)


class ExampleThreeBand(NBandBandstructureDipoleSystem):

    def __init__(self, 
                a = sp.Symbol('a', real=True), 
                prefac_x = sp.Symbol('prefac_x'), 
                prefac_y = sp.Symbol('prefac_y') , flag = None):

        e = []

        e1 = sp.cos(a*self.kx)
        e.append(e1)
        e2 = sp.sin(a*self.ky)
        e.append(e2)
        e3 = sp.cos(2*a*self.kx+self.ky)
        e.append(e3)
        n = 3

        super().__init__(e, prefac_x, prefac_y, n, flag)