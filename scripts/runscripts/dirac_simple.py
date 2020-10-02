import os
import numpy as np
from params import params

import sbe.dipole
import sbe.example
from sbe.solver import sbe_solver


def dirac():
    # Param file adjustments
    # System parameters
    A = 0.19732     # Fermi velocity

    dirac_system = sbe.example.BiTe(C0=0, C2=0, A=A, R=0, mz=0)
    h_sym, ef_sym, wf_sym, _ediff_sym = dirac_system.eigensystem(gidx=1)
    dirac_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)

    return dirac_system, dirac_dipole

def run(system, dipole):

    params.gauge = 'length'
    params.BZ_type = '2line'
    params.Nk_in_path = 100

    params.E0 = 5
    params.w = 25
    params.alpha = 25

    params.e_fermi = 0.0
    params.temperature = 0.0

    stretch_t0 = 5
    # Increase time interval for broader pulses
    if (params.alpha > 25):
        stretch_t0 = 2
    if (params.alpha > 75):
        stretch_t0 = 3

    # Double time for broader pulses
    params.t0 *= stretch_t0
    params.Nt *= stretch_t0

    sbe_solv


if __name__ == "__main__":
    run(*dirac())
