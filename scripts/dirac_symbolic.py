import numpy as np

from hfsbe.example import Dirac
from hfsbe.dipole import SymbolicDipole


def kmat(kinit):
    kinit = np.linspace(-2*np.pi, 2*np.pi, 20)
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    dirac = Dirac(m=0)
    h, ef, wf, ediff = dirac.eigensystem(gidx=1)

    if (eflag):
        dirac.evaluate_energy(kx, ky)
        dirac.plot_bands_3d(kx, ky)
        dirac.plot_bands_contour(kx, ky)
    if (edflag):
        dirac.evaluate_ederivative(kx, ky)
        dirac.plot_bands_derivative(kx, ky)
    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        dip.plot_dipoles(kx, ky)


if __name__ == "__main__":
    N = 20
    kinit = np.linspace(-0.2, 0.2, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True, dipflag=True)
