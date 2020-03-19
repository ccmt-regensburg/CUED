import numpy as np
import matplotlib.pyplot as plt

from hfsbe.example import BiTePeriodic
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature

eV_conv = 0.03674932176
au_conv = (1/eV_conv)


a = 8.308
# C2 = 6.5242
C2 = 1.5242
A = 0.1974
R = 11.06


def energy_line(kx, ky):
    bite = BiTePeriodic(m=1.0, a=a, A=A, C2=C2, R=R, order=4)
    h_sym, ef_sym, wf_sym, ediff_sym = bite.eigensystem()

    ev, ec = bite.evaluate_energy(kx, ky)
    plt.plot(ky, au_conv*ev)
    plt.plot(ky, au_conv*ec)
    plt.ylim(-0.1, 5.0)
    plt.show()


if __name__ == "__main__":
    N = 201
    kx = np.zeros(N)
    ky = np.linspace(-2*np.pi/a, 2*np.pi/a, N)

    energy_line(kx, ky)
