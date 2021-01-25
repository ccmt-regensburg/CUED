import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from sbe.example import BiTeResummed
from sbe.dipole import SymbolicDipole, SymbolicCurvature
from sbe.utility import evaluate_njit_matrix as evmat

np.set_printoptions(linewidth=200)

eV_conv = 0.03674932176
au_conv = (1/eV_conv)


a = 8.28834
C0 = -0.00647156
A = 0.0422927
c2 = 0.0117598
r = 0.109031
ksym = 0.0635012
kasym = 0.113773


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def bite_resummed(kx, ky):
    mb = 0.0003
    bite_mb_sym = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    bite_mb_num = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym, mb=mb)
    h_mb_sym, e_mb_sym, wf_mb_sym, ediff_mb_sym = bite_mb_sym.eigensystem(gidx=1)
    h_mb_num, e_mb_num, wf_mb_num, ediff_mb_num = bite_mb_num.eigensystem(gidx=1)

    print("Hamiltonian")
    print(evmat(bite_mb_num.hfjit, kx=kx, ky=ky))
    print(evmat(bite_mb_sym.hfjit, kx=kx, ky=ky, mb=mb))
    print("Hamiltonian d/dkx")
    print(evmat(bite_mb_num.hderivfjit[0], kx=kx, ky=ky))
    print(evmat(bite_mb_sym.hderivfjit[0], kx=kx, ky=ky, mb=mb))
    print("Hamiltonian d/dky")
    print(evmat(bite_mb_num.hderivfjit[1], kx=kx, ky=ky))
    print(evmat(bite_mb_sym.hderivfjit[1], kx=kx, ky=ky, mb=mb))

    print("Valence Band")
    print(bite_mb_num.efjit[0](kx=kx, ky=ky))
    print(bite_mb_sym.efjit[0](kx=kx, ky=ky, mb=mb))
    print("Conduction Band")
    print(bite_mb_num.efjit[1](kx=kx, ky=ky))
    print(bite_mb_sym.efjit[1](kx=kx, ky=ky, mb=mb))

    print("Valence band d/dkx")
    print(bite_mb_num.ederivfjit[0](kx=kx, ky=ky))
    print(bite_mb_sym.ederivfjit[0](kx=kx, ky=ky, mb=mb))
    print("Valence band d/dky")
    print(bite_mb_num.ederivfjit[1](kx=kx, ky=ky))
    print(bite_mb_sym.ederivfjit[1](kx=kx, ky=ky, mb=mb))
    print("Conduction band d/dkx")
    print(bite_mb_num.ederivfjit[2](kx=kx, ky=ky))
    print(bite_mb_sym.ederivfjit[2](kx=kx, ky=ky, mb=mb))
    print("Conduction band d/dky")
    print(bite_mb_num.ederivfjit[3](kx=kx, ky=ky))
    print(bite_mb_sym.ederivfjit[3](kx=kx, ky=ky, mb=mb))

    dip_mb_num = SymbolicDipole(h_mb_num, e_mb_num, wf_mb_num, offdiagonal_k=True)
    dip_mb_sym = SymbolicDipole(h_mb_sym, e_mb_sym, wf_mb_sym, offdiagonal_k=True)
    print("Dipole Matrix x-Component")
    Ax_mb_num = evmat(dip_mb_num.Axfjit, kx=kx, ky=ky)
    Ax_mb_sym = evmat(dip_mb_sym.Axfjit, kx=kx, ky=ky, mb=mb)
    print(Ax_mb_num)
    print(Ax_mb_sym)
    print("Difference")
    print(Ax_mb_sym - Ax_mb_num)

    print("Dipole Matrix y-Component")
    Ay_mb_num = evmat(dip_mb_num.Ayfjit, kx=kx, ky=ky)
    Ay_mb_sym = evmat(dip_mb_sym.Ayfjit, kx=kx, ky=ky, mb=mb)
    print(Ay_mb_num)
    print(Ay_mb_sym)
    print("Difference")
    print(Ay_mb_sym - Ay_mb_num)

    print("Dipole Matrix x-Component (k neq kp)")
    Ax_mb_num_offk = evmat(dip_mb_num.Axfjit_offk, kx=kx, ky=ky, kxp=kx, kyp=ky)
    Ax_mb_sym_offk = evmat(dip_mb_sym.Axfjit_offk, kx=kx, ky=ky, kxp=kx, kyp=ky, mb=mb)
    print(Ax_mb_num_offk)
    print(Ax_mb_sym_offk)
    print("Difference")
    print(Ax_mb_sym_offk - Ax_mb_num_offk)

    print("Dipole Matrix y-Component (k neq kp)")
    Ay_mb_num_offk = evmat(dip_mb_num.Ayfjit_offk, kx=kx, ky=ky, kxp=kx, kyp=ky)
    Ay_mb_sym_offk = evmat(dip_mb_sym.Ayfjit_offk, kx=kx, ky=ky, kxp=kx, kyp=ky, mb=mb)
    print(Ay_mb_num_offk)
    print(Ay_mb_sym_offk)
    print("Difference")
    print(Ay_mb_sym_offk - Ay_mb_num_offk)

    # cur = SymbolicCurvature(h_sym, dip.Ax, dip.Ay)
    # print("Berry curvature")
    # print(evmat(cur.Bfjit, kx=kx, ky=ky, mb=mb))


if __name__ == "__main__":
    N = 4
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.ones(N)
    bite_resummed(kx, ky)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
