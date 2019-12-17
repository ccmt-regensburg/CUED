import numpy as np

from hfsbe.example import Bite
from hfsbe.dipole import SymbolicDipole
from hfsbe.brillouin import in_brillouin

import plot_dipoles as plt


b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 50)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

inbz = in_brillouin(kx, ky, b1, b2)
kx = kx[inbz]
ky = ky[inbz]

A = 2.8413
R = -3.3765

bite = Bite()
h, ef, wf, ediff = bite.eigensystem()

dip = SymbolicDipole(h, ef, wf)

Ax, Ay = dip.evaluate(kx, ky, b1=b1, b2=b2, A=A, R=R)#, hamiltonian_radius=1.0)

plt.plot_dipoles(kx, ky, Ax, Ay, 'BiTe')
