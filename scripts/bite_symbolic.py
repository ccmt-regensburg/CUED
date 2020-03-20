import sympy as sp
import numpy as np
from numba import njit

from hfsbe.example import BiTe
from hfsbe.example import BiTeTrivial
from hfsbe.dipole import SymbolicDipole
from hfsbe.utility import evaluate_njit_matrix as ev_mat


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def derivtest():
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    h, ef, wf, ediff = bite.eigensystem(gidx=0)

    N = 10
    kxl = np.zeros(N)
    kyl = np.linspace(1, 2*np.pi, N)
    sp.pprint(bite.hderiv[0])
    sp.pprint(bite.hderiv[1])

    hdx = ev_mat(bite.hderivfjit[0], kx=kxl, ky=kyl)
    hdy = ev_mat(bite.hderivfjit[1], kx=kxl, ky=kyl)

    print(hdx[:, :, 0])
    print(hdy[:, :, 0])


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    # bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    h, ef, wf, ediff = bite.eigensystem(gidx=0)

    if (eflag):
        bite.evaluate_energy(kx, ky)
        bite.plot_bands_3d(kx, ky)
        bite.plot_bands_contour(kx, ky)
    if (edflag):
        bite.evaluate_ederivative(kx, ky)
        bite.plot_bands_derivative(kx, ky)
    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
#        print(dip.Axf[0][0](kx=kx, ky=ky))
        Ax, Ay = dip.evaluate(kx, ky)
        print(np.real(Ax))
        dip.plot_dipoles(kx, ky)


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
        # curv = SymbolicCurvature(Ax, Ay)
        # curv.evaluate(kx, ky)
        # curv.plot_curvature_contour(kx, ky)


if __name__ == "__main__":
    N = 10
    # kinit = np.linspace(-0.02, 0.02, N)
    kinit = np.linspace(-0.1, 0.1, N, dtype=np.float)
    kx, ky = kmat(kinit)
    # topological(kx, ky, eflag=False, edflag=False, dipflag=False)
    # trivial(kx, ky, energy=True, ediff=True, dipole=True)
    derivtest()
