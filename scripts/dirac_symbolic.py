import numpy as np

import hfsbe.example as ex
import hfsbe.dipole as dip

import plot_dipoles as plt

# b1 = 2*np.array([1, 0])
# b2 = 2*np.array([0, 1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 20)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

h, ef, wf = ex.TwoBandSystems().dirac()
dipole = dip.SymbolicDipole(h, ef, wf)

m = 0.0
kxbz, kybz, Ax, Ay = dipole.evaluate(kx, ky, m=m)

plt.plot_dipoles(kxbz, kybz, Ax, Ay, 'Dirac')
