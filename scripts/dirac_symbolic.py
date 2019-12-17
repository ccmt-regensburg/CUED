import numpy as np

from hfsbe.example import Dirac
from hfsbe.brillouin import in_brillouin
import hfsbe.dipole as dip

import plot_dipoles as plt

b1 = 2*np.pi*np.array([1, 0])
b2 = 2*np.pi*np.array([0, -1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 20)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

inbz = in_brillouin(kx, ky, b1, b2)
kx = kx[inbz]
ky = ky[inbz]

h, ef, wf, ed = Dirac().eigensystem()
dipole = dip.SymbolicDipole(h, ef, wf)

m = 0.0
Ax, Ay = dipole.evaluate(kx, ky, m=m)

plt.plot_dipoles(kx, ky, Ax, Ay, 'Dirac')
