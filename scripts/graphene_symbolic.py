import numpy as np

from hfsbe.brillouin import in_brillouin
from hfsbe.example import Graphene
import hfsbe.dipole as dip

import plot_dipoles as plt

b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 32)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

inbz = in_brillouin(kx, ky, b1, b2)
kx = kx[inbz]
ky = ky[inbz]

h, ef, wf, ed = Graphene().eigensystem()
dipole = dip.SymbolicDipole(h, ef, wf)

Ax, Ay = dipole.evaluate(kx, ky, t=1, b1=b1, b2=b2)
plt.plot_dipoles(kx, ky, Ax, Ay, 'Graphene')
