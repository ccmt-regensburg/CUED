import numpy as np
import sympy as sp
from params import params

import cued.hamiltonian
from cued.main import sbe_solver

def dirac():
	A = 0.1974      # Fermi velocity

	kx= sp.Symbol('kx', real=True)
	ky = sp.Symbol('ky', real=True)

	dx = +ky / (2 * (kx**2 + ky**2) )
	dy = -kx / (2 * (kx**2 + ky **2) )

	prex = dx*sp.ones(2,2)
	prey = dy*sp.ones(2,2)

	dirac_system = cued.hamiltonian.BiTeBandstructure(vF=A, prefac_x = prex, prefac_y = prey, flag='dipole')

	return dirac_system

def run(system):
	params.solver = 'nband'
	sbe_solver(system, params)

	return 0

if __name__ == "__main__":
	run(dirac())
