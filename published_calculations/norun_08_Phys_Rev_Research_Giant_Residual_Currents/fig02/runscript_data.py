from params_data import params
import numpy as np
from numba import njit

import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as CoFa

def dirac():
        A = 0.1974      # Fermi velocity

        dirac_system = cued.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

        return dirac_system

def single_cycle_pulse(E0, sigma, phi, t0, dt):

        E0 = E0*CoFa.MVpcm_to_au
        sigma = sigma*CoFa.fs_to_au

        @njit
        def electric_field(t):
                return E0 * (2 * t * np.cos( phi )/sigma + ( 1 - 2 * (t/sigma)**2 ) * np.sin( phi ) ) * np.exp( - (t/sigma)**2)
        return electric_field



def run(system):

        params.electric_field_function = single_cycle_pulse(0.5, 50, 0, params.t0, params.dt)
        params.user_defined_header = 'symmetric'
        sbe_solver(system, params)

        params.electric_field_function = single_cycle_pulse(0.5, 50, np.pi/2, params.t0, params.dt)
        params.user_defined_header = 'antisymmetric'
        sbe_solver(system, params)

        return 0

if __name__ == "__main__":

        run(dirac())
