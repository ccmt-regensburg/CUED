import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from hfsbe.example import BiTe
# from hfsbe.dipole import SymbolicDipole

b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

# kinit = np.linspace(-2*np.pi, 2*np.pi, 50)
N = 500
kinit = np.linspace(-0.10, 0.10, N)

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

bite_one = BiTe(A=0, R=1, C0=0, C2=0, vf=1)
h_one, ef_one, wf_one, ediff_one = bite_one.eigensystem(gidx=1)

bite_two = BiTe(A=1, R=1, C0=0, C2=0, vf=0)
h_two, ef_two, wf_two, ediff_two = bite_two.eigensystem(gidx=1)

sp.pprint(ediff_one[0])
sp.pprint(ediff_two[0])


ediff_one_eval = bite_one.evaluate_ederivative(kinit, np.zeros(N))
ediff_two_eval = bite_two.evaluate_ederivative(kinit, np.zeros(N))
plt.plot(kinit, ediff_one_eval[0])
plt.plot(kinit, ediff_two_eval[0])
plt.show()
# bite.evaluate_energy(kx, ky)
# bite.evaluate_ederivative(kx, ky)
# bite.plot_energies_3d(kx, ky)
# bite.plot_energies_contour(kx, ky)
# dip = SymbolicDipole(h, ef, wf)
# Ax, Ay = dip.evaluate(kx, ky)
# print(np.shape(Ax))
# dip.plot_dipoles(kx, ky)

# plt.plot_dipoles(kx, ky, Ax, Ay, 'BiTe')
