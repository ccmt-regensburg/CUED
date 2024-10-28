from params_large_E0_data import params
import numpy as np
from numba import njit

import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as CoFa

def dirac():
        A = 0.1974      # Fermi velocity

        dirac_system = cued.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

        return dirac_system

def run(system):

    sbe_solver(system, params)

    return 0

if __name__ == "__main__":
        run(dirac())

