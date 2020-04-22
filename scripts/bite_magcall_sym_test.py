import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from hfsbe.example import BiTeResummed
from hfsbe.dipole import SymbolicDipole, SymbolicParameterDipole, \
                         SymbolicCurvature
from hfsbe.utility import evaluate_njit_matrix as evmat

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


def bite_resummed_num(kx, ky, eflag=False, edflag=False, dipflag=False):
    mb = 0.0003
    bite = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)
    breakpoint()


if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    bite_resummed_num(kx, ky)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
