import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from hfsbe.example import BiTe
from hfsbe.dipole import SymbolicDipole, SymbolicParameterDipole, \
                         SymbolicCurvature


def dirac_num(kx, ky):

    A = 0.19732
    mb = 0.0003
    dirac = BiTe(C0=0, C2=0, A=A, R=0)
    h_sym, ef_sym, wf_sym, ediff_sym = dirac.eigensystem(gidx=1)


if __name__ == "__main__":
    N = 10
    kinit = np.linspace(-1.0, 1.0, N)
    kx = kinit
    ky = np.zeros(N)
    dirac_num(kx, ky)
