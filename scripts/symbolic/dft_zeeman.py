import os
import numpy as np
import matplotlib.pyplot as plt

from sbe.example import BiTeResummed
from sbe.dipole import SymbolicDipole, SymbolicZeemanDipole
from sbe.utility import conversion_factors as co

def dft():
    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening

    dft_system = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym, zeeman=True)
    h_sym, ef_sym, wf_sym, _ediff_sym = dft_system.eigensystem(gidx=1)
    dft_kdipole = SymbolicDipole(h_sym, ef_sym, wf_sym)
    dft_zdipole = SymbolicZeemanDipole(h_sym, wf_sym)

    return dft_system, dft_kdipole, dft_zdipole

def kmat(kinit):
    kbuf = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
    ky = kbuf[:, 0]
    kx = kbuf[:, 1]
    return kx, ky

def run(system, dipole, zdipole):
    a = 8.28834
    N = 50

    kinit = np.pi*np.linspace(-3/(2*a), 3/(2*a), N)
    kx, ky = kmat(kinit)
    breakpoint()

    B = co.T_to_au * 10
    mu_z = co.muB_to_au * 50
    m_zee_z = B*mu_z

    ky = np.zeros(N)
    ev = system.efjit[0](kx=kx, ky=ky, m_zee_x=0, m_zee_y=0, m_zee_z=m_zee_z)
    ec = system.efjit[1](kx=kx, ky=ky, m_zee_x=0, m_zee_y=0, m_zee_z=m_zee_z)




if __name__ == "__main__":
    run(*dft())
