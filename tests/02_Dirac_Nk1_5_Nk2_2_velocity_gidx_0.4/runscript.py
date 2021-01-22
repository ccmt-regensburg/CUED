import os
import numpy as np
import importlib
from params import params

import sbe.dipole
import sbe.example
from sbe.solver import sbe_solver

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = sbe.example.BiTe(C0=0, C2=0, A=A, R=0, mz=0)
    h_sym, ef_sym, wf_sym, _ediff_sym = dirac_system.eigensystem(gidx=0.4)
    dirac_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    dirac_curvature = sbe.dipole.SymbolicCurvature(h_sym, dirac_dipole.Ax, dirac_dipole.Ay)

    return dirac_system, dirac_dipole, dirac_curvature

def run(system, dipole, curvat):

    sbe_solver(system, dipole, params, curvat)

    return 0

if __name__ == "__main__":
    run(*dirac())
