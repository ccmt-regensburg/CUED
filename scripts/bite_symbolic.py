import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from hfsbe.example import BiTe
from hfsbe.example import BiTeTrivial
# from hfsbe.dipole import SymbolicDipole

b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

# kinit = np.linspace(-2*np.pi, 2*np.pi, 50)
N = 20
kinit = np.linspace(-0.10, 0.10, N)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

# inbz = in_brillouin(kx, ky, b1, b2)
# kx = kx[inbz]
# ky = ky[inbz]

bite_one = BiTe(A=0.1974, R=11.06, C0=0, C2=0)
h_one, ef_one, wf_one, ediff_one = bite_one.eigensystem(gidx=1)

bite_two = BiTeTrivial(R=11.06, C0=0, C2=0, vf=0.1974)
h_two, ef_two, wf_two, ediff_two = bite_two.eigensystem(gidx=1)

one_eval = bite_one.evaluate_ederivative(kx, ky)
two_eval = bite_two.evaluate_ederivative(kx, ky)

bite_one.plot_bands_derivative(kx, ky)
bite_two.plot_bands_derivative(kx, ky)
# bite.evaluate_energy(kx, ky)
# bite.evaluate_ederivative(kx, ky)
# bite.plot_energies_3d(kx, ky)
# bite.plot_energies_contour(kx, ky)
# dip = SymbolicDipole(h, ef, wf)
# Ax, Ay = dip.evaluate(kx, ky)
# print(np.shape(Ax))
# dip.plot_dipoles(kx, ky)

# plt.plot_dipoles(kx, ky, Ax, Ay, 'BiTe')
