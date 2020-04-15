import numpy as np
import matplotlib.pyplot as plt

from hfsbe.example import Dirac
from hfsbe.dipole import SymbolicDipole


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def dirac(kx, ky):
    dirac = Dirac()
    h, ef, wf, ediff = dirac.eigensystem(gidx=1)
    # ev = dirac.efjit[0](kx=kx, ky=ky, vx=1, vy=1, m=0)
    # ec = dirac.efjit[1](kx=kx, ky=ky, vx=1, vy=1, m=0)
    # plt.plot(np.vstack((ev, ec)).T)

    # evdkx = dirac.ederivfjit[0](kx=kx, ky=ky, vx=2, vy=1, m=0)
    # ecdkx = dirac.ederivfjit[2](kx=kx, ky=ky, vx=2, vy=1, m=0)
    print(dirac.hderivfjit[0][0][1](kx=kx, ky=ky, vx=1, vy=1, m=0))

    # dip = SymbolicDipole(h, ef, wf)

if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-1.0, 1.0, N)
    kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    dirac(kx, ky)
