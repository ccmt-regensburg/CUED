import numpy as np
import matplotlib.pyplot as plt

from sbe.example import Dirac
from sbe.dipole import SymbolicDipole


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def dirac(kx, ky):
    dirac = Dirac(vx=1, vy=1)
    h, ef, wf, ediff = dirac.eigensystem(gidx=0.5)
    breakpoint()

if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-1.0, 1.0, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    dirac(kx, ky)
