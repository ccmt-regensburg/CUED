import numpy as np

from hfsbe.example import BiTe
from hfsbe.dipole import SymbolicDipole


b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

# kinit = np.linspace(-2*np.pi, 2*np.pi, 50)
kinit = np.linspace(-0.04, 0.04, 501)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

# inbz = in_brillouin(kx, ky, b1, b2)
# kx = kx[inbz]
# ky = ky[inbz]

A = 0
R = 1
C0 = 0
C2 = 0
vf = 1

bite = BiTe(A=A, R=R, C0=C0, C2=C2, vf=vf)
h, ef, wf, ediff = bite.eigensystem(gidx=1)
# bite.evaluate_energy(kx, ky)
# bite.evaluate_ederivative(kx, ky)
# bite.plot_energies_3d(kx, ky)
# bite.plot_energies_contour(kx, ky)
dip = SymbolicDipole(h, ef, wf)
Ax, Ay = dip.evaluate(kx, ky)
# dip.plot_dipoles(kx, ky)

# plt.plot_dipoles(kx, ky, Ax, Ay, 'BiTe')
