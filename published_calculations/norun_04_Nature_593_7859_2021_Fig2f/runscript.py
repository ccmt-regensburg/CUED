import numpy as np
from params import params

import cued.hamiltonian
from cued.main import sbe_solver

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

def run(system):

    sbe_solver(system, params)

if __name__ == "__main__":
    run(semich_bite())
