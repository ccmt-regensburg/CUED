import numpy as np

import hfsbe.example as ex
import hfsbe.dipole as dip
import hfsbe.utility as utility

import plot_dipoles as plt


# b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
# b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])
b1 = 2*0.04*np.array([1, 0])
b2 = 2*0.04*np.array([0, 1])

kinit = np.linspace(-0.04, 0.04, 20)

# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

A = 2.8413
R = -3.3765

h, ef, wf, ediff = ex.TwoBandSystems(e_deriv=True).bite(A=A, R=R)

evdx = utility.to_numpy_function(ediff[0])
evdy = utility.to_numpy_function(ediff[1])

dipole = dip.SymbolicDipole(h, ef, wf)

kxbz, kybz, Ax, Ay = dipole.evaluate(kx, ky, b1=b1, b2=b2,
                                     hamiltonian_radius=None)

plt.plot_dipoles(kxbz, kybz, Ax, Ay, 'BiTe')
