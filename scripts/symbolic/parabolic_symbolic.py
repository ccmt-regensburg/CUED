import numpy as np
import matplotlib.pyplot as plt

from sbe.example import Parabolic
from sbe.dipole import SymbolicDipole


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def parabolic(kx, ky):
    para = Parabolic(A=1)
    h, ef, wf, ediff = para.eigensystem()

    dip = SymbolicDipole(h, ef, wf)
    breakpoint()


if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-1.0, 1.0, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    parabolic(kx, ky)
