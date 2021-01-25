import os
import numpy as np
from params import params

import sbe.dipole
from sbe.hamiltonian import BiTe
from sbe.utility import mkdir_chdir
from sbe.parameter_loops.sequential import chirp_phasesweep


def bite():
    # Param file adjustments
    # System parameters
    C2 = 5.39018
    A = 0.19732     # Fermi velocity
    R = 5.52658
    k_cut = 0.05

    bite_system = BiTe(C0=0, C2=C2, A=A, R=R, kcut=k_cut, mz=0)
    h_sym, ef_sym, wf_sym, _ediff_sym = bite_system.eigensystem(gidx=1)
    bite_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    bite_curvature = sbe.dipole.SymbolicCurvature(h_sym, bite_dipole.Ax, bite_dipole.Ay)

    return bite_system, bite_dipole, bite_curvature

def run(system, dipole, curvature):

    params.gauge = 'length'
    params.BZ_type = 'hexagon'

    params.E0 = 3
    params.w = 25
    params.alpha = 25

    params.e_fermi = 0.2

    stretch_t0 = 1
    # Increase time interval for broader pulses
    if params.alpha > 25:
        stretch_t0 = 2
    if params.alpha > 75:
        stretch_t0 = 3

    params.t0 *= stretch_t0
    params.dt = 0.1

    Nk1list = [900]
    Nk2list = [120]

    T2list = np.array([1, 2, 3, 5, 7, 10])
    chirplist = [-0.920]
    phaselist = [0.00]

    for Nk1, Nk2 in zip(Nk1list, Nk2list):
        params.Nk1 = Nk1
        params.Nk2 = Nk2
        dirname_dist = 'Nk1_' + '{:d}'.format(Nk1) +  '_Nk2_' + '{:d}'.format(Nk2)
        mkdir_chdir(dirname_dist)

        for T2 in T2list:
            params.T1 = 1000
            params.T2 = T2
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, curvature, params)

            os.chdir('..')
        os.chdir('..')

if __name__ == "__main__":
    run(*bite())
