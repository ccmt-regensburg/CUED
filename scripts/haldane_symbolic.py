import numpy as np

import hfsbe.example as ex
import hfsbe.dipole as dip

import plot_dipoles as plt

# b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
# b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])
b1 = 2*np.pi*np.array([1, 0])
b2 = 2*np.pi*np.array([0, 1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 32)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

h, ef, wf = ex.TwoBandSystems().haldane()
dipole = dip.SymbolicDipole(h, ef, wf)
kxbz, kybz, Ax, Ay = dipole.evaluate(kx, ky, t1=3, t2=1, m=1,
                                     phi=0, b1=b1, b2=b2)
plt.plot_dipoles(kxbz, kybz, Ax, Ay, r'Haldane; $M/t_2=1$ $\phi=0$')
