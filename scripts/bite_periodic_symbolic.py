import numpy as np
import matplotlib.pyplot as plt

from hfsbe.example import BiTePeriodic
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature

eV_conv = 0.03674932176
au_conv = (1/eV_conv)


a = 8.308
A = 0.1974
R = 11.06

def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTePeriodic(m=0.0, a=a, A=A, R=R, order=4)
    h_sym, ef_sym, wf_sym, ediff_sym = bite.eigensystem()

    if (eflag):
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (edflag):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipflag):
        dip = SymbolicDipole(h_sym, ef_sym, wf_sym)
        # curv = SymbolicCurvature(dip.Ax, dip.Ay)
        # for i in range(2000):
        #     print("Round: ", i)
        #     kx = np.random.random_sample(size=400)
        #     ky = np.random.random_sample(size=400)


if __name__ == "__main__":
    N = 2001
    kinit = np.linspace(-np.pi, np.pi, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True, edflag=False, dipflag=False)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
