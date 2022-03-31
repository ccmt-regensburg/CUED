from params import params

import cued.hamiltonian
from cued.main import sbe_solver

def dft():
    t1 = 0.08                 #t coefficient
    lso = 0.0048              #lso coefficient
    lv = 0.03                #lv coefficient
    dft_system = cued.hamiltonian.KM4bandAna(a=params.a, t1=t1, lso=lso, lv=lv)
    return dft_system

def run(system):
    sbe_solver(system, params)
    return 0

if __name__ == "__main__":
    run(dft())
