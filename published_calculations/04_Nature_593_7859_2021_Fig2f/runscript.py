import numpy as np
from params import params

import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as co

def semich_bite():
    # Hamiltonian Parameters
    A = 2*co.eV_to_au

    # Gaps used in the dirac system
    mx = 0.05*co.eV_to_au
    muz = 0.033

    semich_bite_system = cued.hamiltonian.Semiconductor(A=A, mz=muz, mx=mx, a=params.a, nature=True)

    return semich_bite_system

def run(system):

    sbe_solver(system, params)

if __name__ == "__main__":
    run(semich_bite())
