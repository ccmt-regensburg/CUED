import numpy as np
import matplotlib.pyplot as plt

from sbe.example import BiTePeriodic
from sbe.dipole import SymbolicDipole, SymbolicCurvature

eV_conv = 0.03674932176
au_conv = (1/eV_conv)


a = 8.3053
C2 = 5.39018
A = 0.19732
R = 5.52658


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTePeriodic(m=1.0, a=a, A=A, C2=C2, R=R, order=4)
    h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)

    dip = SymbolicDipole(h_sym, e_sym, wf_sym)

    evdx = bite.ederivfjit[0](kx=0.1, ky=0)
    evdy = bite.ederivfjit[1](kx=0.1, ky=0)
    ecdx = bite.ederivfjit[2](kx=0.1, ky=0)
    ecdy = bite.ederivfjit[3](kx=0.1, ky=0)

    di_01x = dip.Axfjit[0][1](kx=0.1, ky=0)
    di_01y = dip.Ayfjit[0][1](kx=0.1, ky=0)

    breakpoint()

    # if (eflag):
    #     ev, ec = bite.evaluate_energy(kx, ky)

        # bite.plot_bands_contour(kx, ky)
        # bite.plot_bands_3d(kx, ky)

    # if (edflag):
        # bite.evaluate_ederivative(kx, ky)
        # bite.plot_bands_derivative(kx, ky)

    # if (dipflag):
        # dip = SymbolicDipole(h_sym, ef_sym, wf_sym)
        # curv = SymbolicCurvature(dip.Ax, dip.Ay)
        # for i in range(2000):
        #     print("Round: ", i)
        #     kx = np.random.random_sample(size=400)
        #     ky = np.random.random_sample(size=400)


if __name__ == "__main__":
    N = 201
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True, edflag=False, dipflag=False)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
