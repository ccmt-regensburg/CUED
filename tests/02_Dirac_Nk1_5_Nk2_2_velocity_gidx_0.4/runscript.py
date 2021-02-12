from params import params

import sbe.hamiltonian
from sbe.main import sbe_solver

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = sbe.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

    return dirac_system
def run(system):

    sbe_solver(system, params, gidx=0.4)

    return 0

if __name__ == "__main__":
    run(dirac())
