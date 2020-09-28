import numpy as np

from hfsbe.example import Graphene
from hfsbe.dipole import SymbolicDipole


b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):
    graphene = Graphene(t=1)
    h, ef, wf, ediff = graphene.eigensystem(gidx=1)

    if (eflag):
        graphene.evaluate_energy(kx, ky)
        graphene.plot_bands_3d(kx, ky)
        graphene.plot_bands_contour(kx, ky)
    if (edflag):
        graphene.evaluate_ederivative(kx, ky)
        graphene.plot_bands_derivative(kx, ky)
    if (dipflag):
        dip = SymbolicDipole(h, ef, wf)
        Ax, Ay = dip.evaluate(kx, ky)
        dip.plot_dipoles(kx, ky)


if __name__ == "__main__":
    N = 40
    kinit = np.linspace(-2*np.pi, 2*np.pi, N)
    kx, ky = kmat(kinit)
    topological(kx, ky, eflag=True, dipflag=True)
