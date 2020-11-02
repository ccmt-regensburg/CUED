import numpy as np

import sbe.dipole
import sbe.example

def kmat(kinit):
    kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    kx = kmat[:, 0]
    ky = kmat[:, 1]
    return kx, ky

def dft():
    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening

    dft_system = sbe.example.BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, ef_sym, wf_sym, _ediff_sym = dft_system.eigensystem(gidx=1)
    dft_dipole = sbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
    dft_curvature = sbe.dipole.SymbolicCurvature(h_sym, dft_dipole.Ax, dft_dipole.Ay)

    return dft_system, dft_dipole, dft_curvature

def check(sys, dip, curv):
    kinit = np.linspace(-0.3, 0.3, 1000)
    kx, ky = kmat(kinit)
    curv.evaluate(kx, ky)
    curv.plot_curvature_contour(kx, ky)


if __name__ == "__main__":
    check(*dft())
