import numpy as np
from numba import njit

from sbe.example import BiTe
from sbe.example import BiTeTrivial
from sbe.dipole import SymbolicDipole


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def compare(kx, ky):
    # bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0, kcut=0.05)
    h_sym, ef_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)

    e = bite.evaluate_energy(kx, ky)
    ed = bite.evaluate_ederivative(kx, ky)
    dip = SymbolicDipole(h_sym, ef_sym, wf_sym)
    Ax, Ay = dip.evaluate(kx, ky)

    evjit = bite.efjit[0](kx=kx, ky=ky)
    ecjit = bite.efjit[1](kx=kx, ky=ky)

    evxjit = bite.ederivjit[0](kx=kx, ky=ky)
    evyjit = bite.ederivjit[1](kx=kx, ky=ky)
    ecxjit = bite.ederivjit[2](kx=kx, ky=ky)
    ecyjit = bite.ederivjit[3](kx=kx, ky=ky)

    di_00x = dip.Axfjit[0][0](kx=kx, ky=ky)
    di_01x = dip.Axfjit[0][1](kx=kx, ky=ky)
    di_11x = dip.Axfjit[1][1](kx=kx, ky=ky)
    di_00y = dip.Ayfjit[0][0](kx=kx, ky=ky)
    di_01y = dip.Ayfjit[0][1](kx=kx, ky=ky)
    di_11y = dip.Ayfjit[1][1](kx=kx, ky=ky)

    ev = e[0] - evjit
    ec = e[1] - ecjit

    evx = ed[0] - evxjit
    evy = ed[1] - evyjit
    ecx = ed[2] - ecxjit
    ecy = ed[3] - ecyjit

    dip00x = Ax[0][0] - di_00x
    dip01x = Ax[0][1] - di_01x
    dip11x = Ax[1][1] - di_11x

    dip00y = Ay[0][0] - di_00y
    dip01y = Ay[0][1] - di_01y
    dip11y = Ay[1][1] - di_11y

    it = [ev, ec, evx, evy, ecx, ecy,
          dip00x, dip01x, dip11x, dip00y, dip01y, dip11y]
    for j in range(10, len(it)):
        for i in range(kx.size):
            print(it[j][i], end='\t')
        print()
#        print(a, b, c, e, f, g, h, i, j, k, l, m)


if __name__ == "__main__":
    N = 50
    # kinit = np.linspace(-0.02, 0.02, N)
    kinit = np.linspace(-0.1, 0.1, N, dtype=np.float)
    kx, ky = kmat(kinit)

    compare(kx, ky)
