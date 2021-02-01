import os
import numpy as np
from params import params

import sbe.dipole
import sbe.hamiltonian
from sbe.parameter_loops.parallel import mkdir_chdir, chirp_phasesweep
from sbe.utility.constants import ConversionFactors as co


def semich_bite():
    # Hamiltonian Parameters
    A = 2*co.eV_to_au

    # Gaps used in the dirac system
    mx = 0.05*co.eV_to_au
    muz = 0.033

    semich_bite_system = sbe.hamiltonian.Semiconductor(A=A, mz=muz, mx=mx,
                                                     a=params.a, align=True)
    h_sym, ef_sym, wf_sym, _ediff_sym = semich_bite_system.eigensystem(gidx=1)
    semich_bite_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    semich_bite_curvature = sbe.dipole.SymbolicCurvature(h_sym, semich_bite_dipole.Ax, semich_bite_dipole.Ay)

    return semich_bite_system, semich_bite_dipole, semich_bite_curvature

def run(system, dipole, curvature):

    params.gauge = 'length'
    params.BZ_type = 'full'

    params.E0 = 2
    params.w = 40
    params.alpha = 25

    params.e_fermi = 0.0

    stretch_t0 = 2
    # Increase time interval for broader pulses
    if (params.alpha > 25):
        stretch_t0 = 2
    if (params.alpha > 75):
        stretch_t0 = 3

    params.t0 *= stretch_t0
    params.Nt *= stretch_t0

    T2list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    chirplist = [-0.920]
    phaselist = np.linspace(0, np.pi, 20)

    params.Nk1 = 1800
    params.Nk2 = 240
    dirname_dist = 'Nk1_{:d}_Nk2_{:d}_nog'.format(params.Nk1, params.Nk2)
    mkdir_chdir(dirname_dist)

    for T2 in T2list:
        pid = os.fork()

        if pid == 0:
            params.T1 = 1000
            params.T2 = T2
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, curvature, params)
            return 0

if __name__ == "__main__":
    run(*semich_bite())
