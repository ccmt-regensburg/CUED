from params import params

import sbe.dipole
import sbe.hamiltonian
from sbe.solver_n_bands import sbe_solver_n_bands

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = sbe.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)
    h_sym, ef_sym, wf_sym, _ediff_sym = dirac_system.eigensystem(gidx=1)
    dirac_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    dirac_curvature = sbe.dipole.SymbolicCurvature(h_sym, dirac_dipole.Ax, dirac_dipole.Ay)

    return dirac_system, dirac_dipole, dirac_curvature

def run(system, dipole, curvat):

    sbe_solver_n_bands(system, dipole, params, curvat)

    return 0

if __name__ == "__main__":
    run(*dirac())
