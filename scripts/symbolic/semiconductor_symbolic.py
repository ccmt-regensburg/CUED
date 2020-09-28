import numpy as np

from sbe.example import Semiconductor
from sbe.dipole import SymbolicDipole


A = 0.19732
mx = 0.027562
a = 4.395


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):

    semic = Semiconductor(A=A, mx=mx, a=a)
    h, ef, wf, ediff = semic.eigensystem(gidx=1)

    if (eflag):
        semic.evaluate_energy(kx, ky)
        semic.plot_bands_3d(kx, ky)
        semic.plot_bands_contour(kx, ky)
    if (edflag):
        semic.evaluate_ederivative(kx, ky)
        semic.plot_bands_derivative(kx, ky)
    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        dip.plot_dipoles(kx, ky)


if __name__ == "__main__":
    N = 40
    kinit = np.linspace(-3/(2*a)*np.pi, 3/(2*a)*np.pi, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True, dipflag=True)
