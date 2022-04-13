from params import params

import cued.hamiltonian
from cued.main import sbe_solver

def dft():
    t = 0.08                 #t coefficient
    dft_system = cued.hamiltonian.Graphene_twoband(a=params.a, t=t)
    return dft_system

def run(system):
    sbe_solver(system, params)
    return 0

if __name__ == "__main__":
    run(dft())
