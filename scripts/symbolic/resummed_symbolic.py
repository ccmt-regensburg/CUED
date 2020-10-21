import numpy as np
import matplotlib.pyplot as plt

from sbe.utility import conversion_factors as co
from sbe.example import BiTeResummed
from sbe.dipole import SymbolicDipole

a = 8.28834
C0 = -0.00647156
A = 0.0422927
c2 = 0.0117598
r = 0.109031
ksym = 0.0635012
kasym = 0.113773

plt.rcParams['figure.figsize'] = (14, 10)


def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def dft_func(kx, ky):
    dft = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, e_sym, wf_sym, ediff_sym = dft.eigensystem(gidx=1)
    dft_dip = SymbolicDipole(h_sym, e_sym, wf_sym)
    dft_dip.evaluate(kx, ky)
    dft_dip.plot_dipoles(co.as_to_au*kx, co.as_to_au*ky, xlabel=r'$k_x [\si{1/\angstrom}]$',
                         ylabel=r'$k_y [\si{1/\angstrom}]$', savename='dft_dipoles.png')


if __name__ == "__main__":
    N = 26
    kinit = np.linspace(-3*np.pi/(2*a), 3*np.pi/(2*a), N)
    kx, ky = kmat(kinit)
    dft_func(kx, ky)
