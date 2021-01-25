from params import params

import sbe.dipole
import sbe.hamiltonian
from sbe.solver import sbe_solver

def dft():
    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening

    dft_system = sbe.hamiltonian.BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, ef_sym, wf_sym, _ediff_sym = dft_system.eigensystem(gidx=1)
    dft_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    dft_curvature = sbe.dipole.SymbolicCurvature(h_sym, dft_dipole.Ax, dft_dipole.Ay)

    return dft_system, dft_dipole, dft_curvature

def run(system, dipole, curvat):

    sbe_solver(system, dipole, params, curvat)

    return 0

if __name__ == "__main__":
    run(*dft())
