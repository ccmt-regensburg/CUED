import numpy as np
import sympy as sp
from params import params

import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as CoFa

def dirac():

    a    = 1.0/params.length_BZ_E_dir*2.0*np.pi
    d0   = 3*CoFa.as_to_au
    t    = 0.5*CoFa.eV_to_au
    eps0 = 1*CoFa.eV_to_au

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    ev=t*sp.cos(kx*a)-eps0+1.0E-6*ky
    ec=-ev

    dipx = d0*sp.cos(kx*a)**2*sp.ones(2,2)
    dipy = d0*sp.cos(kx*a)**2*sp.ones(2,2)

    dirac_system = cued.hamiltonian.fully_flexible_bandstructure_dipoles(ev=ev,ec=ec, dipole_x = dipx, dipole_y = dipy, flag='dipole')

    return dirac_system

def run(system):
    params.solver = 'nband'
    sbe_solver(system, params)

    return 0

if __name__ == "__main__":
    run(dirac())
