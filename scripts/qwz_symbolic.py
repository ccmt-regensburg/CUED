import numpy as np
import sympy as sp

import hfsbe.example as ex
import hfsbe.dipole as dip

import plot_dipoles as plt

# b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
# b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])
# b1 = 2*np.pi*np.array([1, 0])
# b2 = 2*np.pi*np.array([0, 1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 20)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

systems = ex.TwoBandSystems()
h, ef, wf = systems.qwz()
h_t, ef_t, wf_t = systems.qwz(order=1)
sp.pprint(wf_t[0][0, :])

dipole = dip.SymbolicDipole(h_t, ef_t, wf_t)

# kx = sp.Symbol('kx')
# ky = sp.Symbol('ky')
# sp.pprint(sp.diff(wf[0][0, :], kx))
# sp.pprint(sp.diff(wf_t[0][0, :], kx))

m = 2
kxbz, kybz, Ax, Ay = dipole.evaluate(kx, ky, m=m)
plt.plot_dipoles(kxbz, kybz, Ax, Ay, 'Qi-Wu-Zhang; $m=' + str(m) + '$')
