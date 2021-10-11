from params import params

import cued.hamiltonian
from cued.main import sbe_solver

def dft():
	C0 = -0.00647156                  # C0
	c2 = 0.0117598                    # k^2 coefficient
	A = 0.0422927                     # Fermi velocity
	r = 0.109031                      # k^3 coefficient
	ksym = 0.0635012                  # k^2 coefficent dampening
	kasym = 0.113773                  # k^3 coeffcient dampening

	dft_system = cued.hamiltonian.BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)

	return dft_system
def run(system):

	sbe_solver(system, params)

	return 0

if __name__ == "__main__":
	run(dft())
