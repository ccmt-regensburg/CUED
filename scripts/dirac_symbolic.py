import numpy as np
import sympy as sp

from hfsbe.example import Dirac
from hfsbe.brillouin import in_brillouin
import hfsbe.dipole as dip

# import plot_dipoles as plt

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

h, ef, wf1, ed = Dirac(m=0).eigensystem(gidx=None)
dipole_1 = dip.SymbolicDipole(h, ef, wf1)

h, ef, wf2, ed = Dirac(m=0).eigensystem(gidx=0)
dipole_2 = dip.SymbolicDipole(h, ef, wf1)

h, ef, wf3, ed = Dirac(m=0).eigensystem(gidx=1)
dipole_3 = dip.SymbolicDipole(h, ef, wf1)


# Here we compare the expressions for the y direction dipole moment
# For m = 0 and gidx=0 and gidx=1 conduction and valence band should
# have the same dipole moment
sp.pprint(sp.simplify(dipole_1.Ay[0, 0]))
sp.pprint(sp.simplify(dipole_1.Ay[1, 1]))
sp.pprint(sp.simplify(dipole_2.Ay[0, 0]))
sp.pprint(sp.simplify(dipole_2.Ay[1, 1]))
sp.pprint(sp.simplify(dipole_3.Ay[0, 0]))
sp.pprint(sp.simplify(dipole_3.Ay[1, 1]))
# m = 0.0
# Ax, Ay = dipole.evaluate(kx, ky, m=m)

# plt.plot_dipoles(kx, ky, Ax, Ay, 'Dirac')
