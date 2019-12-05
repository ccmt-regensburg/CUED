import numpy as np
import matplotlib.pyplot as plt

import hfsbe.example as ex
import hfsbe.dipole as dip

b1 = 2*np.pi*np.array([1/np.sqrt(3), 1])
b2 = 2*np.pi*np.array([1/np.sqrt(3), -1])

kinit = np.linspace(-2*np.pi, 2*np.pi, 51)
    
# Full kgrid
kmat = np.array(np.meshgrid(kinit, kinit)).T.reshape(-1, 2)
kx = kmat[:, 0]
ky = kmat[:, 1]

h, ef, wf = ex.TwoBandSystems().graphene()
dipole = dip.SymbolicDipole(h, ef, wf)
kxbz, kybz, Ax, Ay = dipole.evaluate(kx, ky, t=1)

# Plot real and imaginary part of all four fields 00, 01, 10, 11
fig, ax = plt.subplots(4, 2)
for i in range(4):
    idx = np.unravel_index(i, (2, 2), order='C')
    ax[i, 0].quiver(kxbz, kybz, np.real(Ax[idx]), np.real(Ay[idx]),
                    angles='xy')
    ax[i, 1].quiver(kxbz, kybz, np.imag(Ax[idx]), np.imag(Ay[idx]),
                    angles='xy')
    plt.show()
