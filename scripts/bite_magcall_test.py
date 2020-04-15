import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from hfsbe.example import BiTeResummed
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature
from hfsbe.utility import evaluate_njit_matrix

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


def bite_resummed(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTeResummed(c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)

    evf, ecf = bite.efjit
    C0 = 0.5
    mb = 0.0003
    ev = evf(C0=C0, kx=kx, ky=ky, mb=mb)
    ec = ecf(C0=C0, kx=kx, ky=ky, mb=mb)

    plt.plot(np.vstack((ev, ec)).T)

    evdkx = bite.ederivfjit[0](C0=C0, kx=kx, ky=ky, mb=0.0003)
    ecdkx = bite.ederivfjit[2](C0=C0, kx=kx, ky=ky, mb=0.0003)
    plt.plot(np.vstack((evdkx, ecdkx)).T)
    plt.show()

    dip = SymbolicDipole(h_sym, e_sym, wf_sym, offdiagonal_k=True)
    Ax = evaluate_njit_matrix(dip.Axfjit, kx=kx, ky=ky, mb=0.0003)
    # di_01y = dip.Ayfjit[0][1](kx=kx, ky=ky, mb=0.0003)

    Ax_offk = evaluate_njit_matrix(dip.Axfjit_offk, kx=kx, ky=ky, kxp=kx, kyp=ky, mb=0.0003)
    print(Ax[0, 1])
    print(Ax_offk[0, 1])


    plt.show()


if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    bite_resummed(kx, ky)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
