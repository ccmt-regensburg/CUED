import os
import numpy as np
import sympy as sp
from params import params

import cued.hamiltonian
from cued.main import sbe_solver


def dirac():
    # Param file adjustments
    # System parameters

    # flags: 
    #   'dipole': give symbolic dipole as prefac_x and prefac_y
    #   'prefac': give dipole(0)*gap(0) as prefac_x and prefac_y
    #   'd0'    : give dipole(0) as prefac_x and prefac_y

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    dx = -ky / (2 * (kx**2 + ky**2) )
    dy = kx / (2 * (kx**2 + ky **2) )

    prex = dx*sp.ones(2,2)
    prey = dy*sp.ones(2,2)

    example_system = cued.hamiltonian.DiracBandstructure(vF=0.1974, prefac_x = prex, prefac_y = prey, flag='dipole')

    return example_system
    
def run(system):

    params.gauge = 'length'
    params.BZ_type = 'rectangle'
    params.hamiltonian_evaluation = 'bandstructure'
    params.solver = 'nband'

    sbe_solver(system, params)

if __name__ == "__main__":
    run(dirac())
