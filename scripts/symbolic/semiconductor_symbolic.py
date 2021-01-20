import numpy as np
import matplotlib.pyplot as plt

from sbe.example import Semiconductor
from sbe.dipole import SymbolicDipole
from sbe.utility import conversion_factors as co

# Hamiltonian Parameters
A = 2*co.eV_to_au
a = 8.28834

# Gaps used in the dirac system
mx = 0.05*co.eV_to_au
muz = 0.033

def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky


def topological(kx, ky, eflag=False, edflag=False, dipflag=False):

    semic = Semiconductor(A=A, mz=muz, mx=mx, a=a, align=True)
    h, ef, wf, ediff = semic.eigensystem(gidx=0)

    if (eflag):
        semic.evaluate_energy(kx, ky)
        semic.plot_bands_3d(kx, ky)
        semic.plot_bands_contour(kx, ky)
    if (edflag):
        semic.evaluate_ederivative(kx, ky)
        semic.plot_bands_derivative(kx, ky)
    if (dipflag):
        dip_kdotp = SymbolicDipole(h, ef, wf, kdotp=[1j*2.5e-2, 0])
        dip_kfull = SymbolicDipole(h, ef, wf)
        Ax_kdotp, Ay_kdotp = dip_kdotp.evaluate(kx, ky)
        Ax_kfull, Ay_kfull = dip_kfull.evaluate(kx, ky)
        plt.plot(kx, Ax_kdotp[0, 1].imag, kx, Ax_kfull[0, 1].imag, ls='', marker='.')
        plt.legend([r'$k\cdot p$ fit', r'$k$-derivative'])
        plt.xlabel(r'$k_x$ in at.u.')
        plt.ylabel(r'$d_{vc}^x$ in at.u.')
        plt.show()
        # dip.plot_dipoles(kx, ky)


if __name__ == "__main__":
    N = 201
    kinit = np.linspace(-3/(2*a)*np.pi, 3/(2*a)*np.pi, N)
    # kx, ky = kmat(kinit)
    kx = kinit
    ky = np.zeros(N)
    topological(kx, ky, eflag=False, dipflag=True)
