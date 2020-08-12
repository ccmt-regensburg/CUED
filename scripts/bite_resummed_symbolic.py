import numpy as np
import matplotlib.pyplot as plt

from hfsbe.example import BiTeResummed
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature

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
    bite = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)

    breakpoint()
    # print(bite.ederivfjit[1](kx=kx, ky=ky) - bite.ederivfjit[1](kx=kx, ky=-ky))


if __name__ == "__main__":
    N = 81
    kx = np.linspace(-3*np.pi/(2*a), 3*np.pi/(2*a), N)
    ky = (0.03*2*np.pi/a)*np.ones(N)
    # kx, ky = kmat(kinit)
    bite_resummed(kx, ky)

