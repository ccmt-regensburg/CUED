import numpy as np
import sympy as sp
from params import params

import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as CoFa


def dirac():

        a                           = 3*CoFa.as_to_au       # Lattice spacing in atomic units, file internal
        params.length_BZ_E_dir      = 2*np.pi/a
        #Energetic parameters
        b                           = 0.15/2 * CoFa.eV_to_au
        c                           = 6*0.3625 * CoFa.eV_to_au
        #Constant dipole
        d0                          = 3*CoFa.as_to_au

        kx = sp.Symbol('kx', real=True)
        ky = sp.Symbol('ky', real=True)

        hz = b + c * (1 - sp.cos(kx*a))

        ev = -np.absolute(hz) + 1e-8*ky
        ec = -ev

        dipx = d0*sp.Matrix([[0,1],[1,0]])
        dipy = 0*sp.ones(2,2)

        dirac_system = cued.hamiltonian.fully_flexible_bandstructure_dipoles(ev=ev,ec=ec, dipole_x = dipx, dipole_y = dipy, flag='dipole')

        return dirac_system

def run(system):
        sbe_solver(system, params)

        return 0

if __name__ == "__main__":
        run(dirac())
