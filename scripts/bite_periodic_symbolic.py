import numpy as np

from hfsbe.example import BiTePeriodic
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTePeriodic(A=0.1974, R=11.06, C0=0, C2=0, a=20)
    h, ef, wf, ediff = bite.eigensystem()

    if (eflag):
        bite.evaluate_energy(kx, ky)
        # bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (edflag):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        curv = SymbolicCurvature(dip.Ax, dip.Ay)
        for i in range(2000):
            print("Round: ", i)
            kx = np.random.random_sample(size=400)
            ky = np.random.random_sample(size=400)
            ed = bite.evaluate_ederivative(kx, ky)
            b = curv.evaluate(kx, ky)
            b[0]
            ed[0]


if __name__ == "__main__":
    N = 201
    kinit = np.linspace(-0.05, 0.05, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
