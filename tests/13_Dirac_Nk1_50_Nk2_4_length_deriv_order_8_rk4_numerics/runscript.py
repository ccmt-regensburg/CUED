from params import params

import sbe.hamiltonian
from sbe.solver import sbe_solver

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = sbe.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

    return dirac_system

def run(system):

    sbe_solver(system, params)

    return 0

if __name__ == "__main__":
    run(dirac())