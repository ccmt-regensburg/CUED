from params import params

import cued.hamiltonian
from cued.main import sbe_solver

def dirac():
	A  = 0.1974      # Fermi velocity
	mz = 0.01837     # prefactor of sigma_z in Hamiltonian
	#mz = 0

	dirac_system = cued.hamiltonian.BiTe_num(C0=0, C2=0, A=A, R=0, mz=mz)

	return dirac_system

def run(system):

	sbe_solver(system, params)

	return 0

if __name__ == "__main__":
	run(dirac())
