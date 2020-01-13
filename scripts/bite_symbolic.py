import numpy as np

from hfsbe.example import BiTe
from hfsbe.dipole import SymbolicDipole


b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

# kinit = np.linspace(-2*np.pi, 2*np.pi, 50)
kinit = np.linspace(-0.04, 0.04, 21)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

# inbz = in_brillouin(kx, ky, b1, b2)
# kx = kx[inbz]
# ky = ky[inbz]

# A = 0.1974
# R = 11.06
# C0 = -0.008269
# C2 = 6.5242
A = 0
R = 0
C0 = 0
C2 = 0

bite = BiTe(A=0, R=0, C0=0, C2=0, vf=1, kcut=0.05)
h, ef, wf, ediff = bite.eigensystem(gidx=1)
bite.evaluate_energy(kx, ky)
bite.evaluate_ederivative(kx, ky)
bite.plot_energies_3d(kx, ky)
# dip = SymbolicDipole(h, ef, wf, b1=b1, b2=b2)
# Ax, Ay = dip.evaluate(kx, ky)
# dip.plot_dipoles(kx, ky)

# plt.plot_dipoles(kx, ky, Ax, Ay, 'BiTe')
