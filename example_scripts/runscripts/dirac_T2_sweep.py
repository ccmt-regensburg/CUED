import os
import numpy as np
from params import params

import sbe.dipole
import sbe.example
from sbe.solver.runloops import mkdir_chdir, chirp_phasesweep


def dirac():
    # Param file adjustments
    # System parameters
    # A = 0.19732     # Fermi velocity

    # For testing purposes against Jans code
    A = 0.1974

    dirac_system = sbe.example.BiTe(C0=0, C2=0, A=A, R=0, mz=0)
    h_sym, ef_sym, wf_sym, _ediff_sym = dirac_system.eigensystem(gidx=1)
    dirac_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    dirac_curvature = sbe.dipole.SymbolicCurvature(h_sym, dirac_dipole.Ax, dirac_dipole.Ay)

    return dirac_system, dirac_dipole, dirac_curvature

def run(system, dipole, curvature):

    params.gauge = 'length'
    params.BZ_type = '2line'
    params.Nk_in_path = 1600

    # For testing purposes against Jans code
    params.a = 8.308

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

    params.length_path_in_BZ = 1600*0.00306
    params.num_paths = 100
    distlist = [0.002]
    T1list = [1000]
    chirplist = [0.000]
    phaselist = np.linspace(0, np.pi, 20)[:1]

    for dist in distlist:
        params.rel_dist_to_Gamma = dist
        dirname_dist = 'dist_{:.2f}_Nk_in_path_{:d}_num_paths_{:d}'.format(dist, params.Nk_in_path, params.num_paths)
        mkdir_chdir(dirname_dist)

        for T1 in T1list:
            params.T1 = T1
            params.T2 = 1
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, curvature, params)

            os.chdir('..')
        os.chdir('..')

if __name__ == "__main__":
    run(*dirac())
