from numba import njit
import numpy as np

from sbe.example import BiTe
from sbe.dipole import SymbolicDipole
np.set_printoptions(linewidth=200)

eV_conv = 0.03674932176
au_conv = (1/eV_conv)


a = 8.308
C0 = 0
C2 = 0
A = 0.1974
R = 5.53
k_cut = 0.05

bite = BiTe(C0=C0, C2=C2, A=A, R=R, kcut=k_cut)
h_sym, e_sym, wf_sym, ediff_sym = bite.eigensystem(gidx=1)
dipole = SymbolicDipole(h_sym, e_sym, wf_sym, offdiagonal_k=True)

di_00xjit = dipole.Axfjit[0][0]
di_01xjit = dipole.Axfjit[0][1]
di_01xjit_offk = dipole.Axfjit_offk[0][1]
di_11xjit = dipole.Axfjit[1][1]

di_00yjit = dipole.Ayfjit[0][0]
di_01yjit = dipole.Ayfjit[0][1]
di_01yjit_offk = dipole.Ayfjit_offk[0][1]
di_11yjit = dipole.Ayfjit[1][1]


@njit
def test(kx, ky, E_dir, dipole_in_path):

    # E_dir[0]*di_00xjit(kx=kx, ky=ky)
    # E_dir[1]*di_01xjit(kx=kx, ky=ky)
    # E_dir[0]*di_11xjit(kx=kx, ky=ky)

    # E_dir[0]*di_00yjit(kx=kx, ky=ky)
    # E_dir[1]*di_01yjit(kx=kx, ky=ky)
    # E_dir[0]*di_11yjit(kx=kx, ky=ky)

    di_01x_B_field = di_01xjit_offk(kx=kx, ky=ky, kxp=ky, kyp=kx)
    di_01y_B_field = di_01yjit_offk(kx=kx, ky=ky, kxp=ky, kyp=kx)
    dipole_in_path = E_dir[0]*di_01x_B_field + E_dir[1]*di_01y_B_field

    return dipole_in_path


if __name__ == "__main__":
    N = 1000
    kinit = np.linspace(-np.pi/a, np.pi/a, N)
    kx = kinit
    ky = np.zeros(N)
    E_dir = np.array([np.cos(0), np.sin(0)])
    di_x, di_y = dipole.evaluate(kx, ky)

    dipole_in_path = E_dir[0]*di_x[0, 1, :] + E_dir[1]*di_y[0, 1, :]

    print(test(kx, ky, E_dir, dipole_in_path))
