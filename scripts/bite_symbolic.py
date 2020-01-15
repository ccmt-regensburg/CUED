import numpy as np

import matplotlib.pyplot as plt

from hfsbe.example import BiTe
from hfsbe.example import BiTeTrivial
from hfsbe.dipole import SymbolicDipole


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, energy=False, ediff=False, dipole=False):
    kx, ky = kmat(kinit)
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    h, ef, wf, ediff = bite.eigensystem(gidx=1)

    if (energy):
        print("Hello")
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (ediff):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipole):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        print(Ay)
        dip.plot_dipoles(kx, ky)


def trivial(kx, ky, energy=False, ediff=False, dipole=False):
    kx, ky = kmat(kinit)
    bite = BiTeTrivial(R=11.06, C0=0, C2=0, vf=0.1974)
    h, ef, wf, ediff = bite.eigensystem(gidx=1)

    if (energy):
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (ediff):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipole):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        print(Ax)
        dip.plot_dipoles(kx, ky)


if __name__ == "__main__":
    N = 20
    kinit = np.linspace(-0.10, 0.10, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, dipole=True)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
