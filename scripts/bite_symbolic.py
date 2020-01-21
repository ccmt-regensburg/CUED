import numpy as np

from hfsbe.example import BiTe
from hfsbe.example import BiTeTrivial
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    h, ef, wf, ediff = bite.eigensystem()

    if (eflag):
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (edflag):
        print(ediff)
        print("Hello World")
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        curv = SymbolicCurvature(dip.Ax, dip.Ay)
        for i in range(2000):
            print("Round: ", i)
            kx = i*0.0001*np.ones(100)
            ky = np.zeros(100)
            b = curv.evaluate(kx, ky)
            b[0]


def trivial(kx, ky, eflag=False, edflag=False, dipflag=False):
    bite = BiTeTrivial(R=11.06, C0=0, C2=0, vf=0.1974)
    h, ef, wf, ediff = bite.eigensystem(gidx=1)

    if (eflag):
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)

    if (edflag):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)

    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        dip.plot_dipoles(kx, ky)
        curv = SymbolicCurvature(Ax, Ay)
        curv.plot_curvature_contour(kx, ky)


if __name__ == "__main__":
    N = 21
    kinit = np.linspace(-0.10, 0.10, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, dipflag=True)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
