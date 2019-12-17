import numpy as np

from hfsbe.example import Qwz
import hfsbe.dipole as dip

import plot_dipoles as plt

b1 = 2*np.pi*np.array([1, 0])
b2 = 2*np.pi*np.array([0, 1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 20)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

h, ef, wf, ed = Qwz().eigensystem()
h_t, ef_t, wf_t, ed_t = Qwz(order=1).eigensystem()

dipole = dip.SymbolicDipole(h_t, ef_t, wf_t)

m = 2
Ax, Ay = dipole.evaluate(kx, ky, m=m)
plt.plot_dipoles(kx, ky, Ax, Ay, 'Qi-Wu-Zhang; $m=' + str(m) + '$')
