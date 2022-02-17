import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct

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
                 mz=sp.Symbol('mz'), a=sp.Symbol('a'), nature=False):
        ho = 0
        hx = mx
        hy = 0

        if (nature):
            hz = A*(2 + mz - sp.cos((2*a/3)*self.kx) - sp.cos((2*a/3)*self.ky))
        else:
            hz = A*(2 + mz - sp.cos((3*a/4)*self.kx) - sp.cos((3*a/4)*self.ky))

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

class BiTeResummed_num(TwoBandHamiltonianSystem):
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

class BiTe_num_4_bands(NBandHamiltonianSystem):
    '''
        Artificial 3Band model with Dirac cone for first two bands, zero else
    '''

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0):

        so = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        sx = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        sy = sp.Matrix([[0, -sp.I, 0, 0], [sp.I, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        sz = sp.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

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

class DiracConeNSheets(NBandHamiltonianSystem):

    '''
       BiTe Hamiltonian on n sheets
    '''

    def __init__(self, C0=sp.Symbol('C0', real=True),
                 C2=sp.Symbol('C2', real=True),
                 A=sp.Symbol('A', real=True),
                 R=sp.Symbol('R', real=True),
                 kcut=0, mz=0,
                 n_sheets = sp.Symbol('n_sheets', int=True)):

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
        hdiag = ho*so + hx*sx + hy*sy + hz*sz

        #build hamiltonian with nz sheets
        diag = sp.eye(n_sheets)
        h = TensorProduct(diag, hdiag)

        super().__init__(h, n_sheets)

class DiracConeAna(NBandHamiltonianSystem):

    def __init__(self, vF=sp.Symbol('vF', real=True)):

        h = vF * sp.Matrix([[0, self.kx - sp.I * self.ky],[self.kx + sp.I * self.ky, 0]])
        ana_e = sp.Matrix([vF*(self.kx**2 + self.ky**2)**(1/2), -vF*(self.kx**2 + self.ky**2)**(1/2)])
        #ana_wf = sp.Matrix([[ (self.kx - sp.I * self.ky)/((2*self.kx**2 + 2*self.ky**2)**(1/2)),  - (self.kx - sp.I * self.ky)/((2*self.kx**2 + 2*self.ky**2)**(1/2))], [2**(-1/2), 2**(-1/2)]])
        ana_wf = sp.Matrix([[ (self.kx - sp.I * self.ky)/((self.kx**2 + self.ky**2)**(1/2)),  - (self.kx - sp.I * self.ky)/((self.kx**2 + self.ky**2)**(1/2))], [ 1 , 1 ]])
        super().__init__(h, 1, False, ana_e, ana_wf)

class AIIIHamiltonian(NBandHamiltonianSystem):

    def __init__(self,
                A = sp.Symbol('A', real=True),
                t = sp.Symbol('t', real=True),
                tt = sp.Symbol('tt', real=True),
                m5 = sp.Symbol('m5', real= True),
                n_sheets = sp.Symbol('n_sheets', int=True)):

        #set of hermitian matrices
        Gamma1=sp.Matrix([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        Gamma2=sp.Matrix([[0,0,0,-sp.I],[0,0,sp.I,0],[0,-sp.I,0,0],[sp.I,0,0,0]])
        Gamma3=sp.Matrix([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]])
        Gamma5=sp.Matrix([[0,0,-sp.I,0],[0,0,0,-sp.I],[sp.I,0,0,0],[0,sp.I,0,0]])

        #diagonal part of hamiltonian
        hdiag = A * ( m5 + t * ( sp.cos(self.kx) + sp.cos(self.ky) ) ) * Gamma5 \
               + A * tt * ( sp.sin(self.kx) * Gamma1 + sp.sin(self.ky) * Gamma2 )

        #hopping between sheets
        hoffdiag = A * ( 1/2*t*Gamma5 + 1/(2*sp.I)*tt*Gamma3 ) #lower offdiag blocks
        hoffdiag_c = A * ( 1/2*t*Gamma5 - 1/(2*sp.I)*tt*Gamma3 ) #upper offdiag blocks

        #build hamiltonian with nz sheets
        diag = sp.eye(n_sheets)
        lower = sp.Matrix.jordan_block(n_sheets, 0, band='lower')
        upper = sp.Matrix.jordan_block(n_sheets, 0, band='upper')

        h = TensorProduct(diag, hdiag) + TensorProduct(lower, hoffdiag) + TensorProduct(upper, hoffdiag_c)

        super().__init__(h, n_sheets, degenerate_eigenvalues=True)


############################################################################################
# Bandstructures and dipoles for kp-evaluation
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

class fully_flexible_bandstructure_dipoles(NBandBandstructureDipoleSystem):

    def __init__(self,
                 ev = sp.Symbol('ev', real=True),
                 ec = sp.Symbol('ec', real=True),
                 dipole_x = sp.Symbol('dipole_x'),
                 dipole_y = sp.Symbol('dipole_y'), flag = 'dipole'):

        e = [ev, ec]
        n = 2

        super().__init__(e, dipole_x, dipole_y, n, flag)

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
