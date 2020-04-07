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


def bite_resummed(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)

    # dip = SymbolicDipole(h_sym, e_sym, wf_sym)

    if (eflag):
        ev, ec = bite.evaluate_energy(kx, ky)

        bite.plot_bands_contour(kx, ky)
        # bite.plot_bands_3d(kx, ky)

    # if (edflag):
        # bite.evaluate_ederivative(kx, ky)
        # bite.plot_bands_derivative(kx, ky)

    if (dipflag):
        dip = SymbolicDipole(h_sym, e_sym, wf_sym)
        curv = SymbolicCurvature(dip.Ax, dip.Ay)
        breakpoint()
#        curv.Bfjit[0][1](kx=kx, ky=ky)
        # for i in range(2000):
        #     print("Round: ", i)
        #     kx = np.random.random_sample(size=400)
        #     ky = np.random.random_sample(size=400)


if __name__ == "__main__":
    N = 200
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx, ky = kmat(kinit)
    bite_resummed(kx, ky, eflag=True, edflag=False, dipflag=True)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
