import numpy as np

from hfsbe.example import BiTe
from hfsbe.example import BiTeTrivial
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, energy=False, ediff=False, dipole=False):
    kx, ky = kmat(kinit)
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    h, ef, wf, ediff = bite.eigensystem()

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
        dip.plot_dipoles(kx, ky)
        curv = SymbolicCurvature(dip.Ax, dip.Ay)
        curv.evaluate(kx, ky)
        curv.plot_curvature_contour(kx, ky)


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
        dip.plot_dipoles(kx, ky)
        curv = SymbolicCurvature(Ax, Ay)
        curv.plot_curvature_contour(kx, ky)


if __name__ == "__main__":
    N = 200
    kinit = np.linspace(-0.10, 0.10, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, energy=True)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
