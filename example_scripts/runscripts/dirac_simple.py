import os
import numpy as np
from params import params

import sbe.dipole
import sbe.hamiltonian
from sbe.main import sbe_solver


def dirac():
    # Param file adjustments
    # System parameters
    A = 0.19732     # Fermi velocity

    dirac_system = sbe.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

    return dirac_system
    
def run(system):

    params.gauge = 'length'
    params.BZ_type = 'rectangle'
    params.system = 'ana'
    params.solver = '2band'

    params.Nk1 = 1080
    params.Nk2 = 2

    params.E0 = 5
    params.w = 25
    params.alpha = 25

    params.e_fermi = 0.0
    params.temperature = 0.0

    sbe_solver(system, params)

if __name__ == "__main__":
    run(dirac())
