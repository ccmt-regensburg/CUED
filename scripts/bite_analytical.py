from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from hfsbe import dipole


plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20

## Bi2Te3 material constants ##
## The constants are the 'tilde' constants from the paper
A = 2.87 * 0.99
C0 = -0.18 + 0.15 * 0.30
C2 = 49.68 - 0.15 * 57.38
R = 45.02 * 0.99/2
####

def hamiltonian(kx, ky):
    so = np.array([[1, 0], [0, 1]], dtype=np.complex)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex)

    H = C0 * so + C2 * (kx**2 + ky**2) * so + A * (sx * ky - sy * kx) + \
        2 * R * (kx**3 - 3 * kx * ky**2) * sz

    return H

def hderivative(kx, ky):
    so = np.array([[1, 0], [0, 1]], dtype=np.complex)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex)

    dxH = 2*C2*kx*so - A*sy +  6*R*(kx**2-ky**2)*sz
    dyH = 2*C2*ky*so + A*sx -  12*R*kx*ky*sz

    return dxH, dyH

if __name__ == "__main__":

    # Same spacing for kx and ky used here, important for derivatives
    kxlist = np.linspace(-0.04, 0.04, 21)
    kylist = kxlist

    bite_dipole = dipole.Dipole(hamiltonian, kxlist, kylist, hderivative=hderivative)

    fig = plt.figure()
    kxv, kyv = np.meshgrid(kxlist, kylist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kxv, kyv, bite_dipole.e[:, :, 0])
    ax.plot_surface(kxv, kyv, bite_dipole.e[:, :, 1])
    plt.show()

    sp = bite_dipole.dipole_field(0, 1, energy_eps=10e-10)

    fig, ax = plt.subplots(2, 1)
    ax[0].quiver(kxlist, kylist, np.real(sp[:, :, 0]), np.real(sp[:, :, 1]),
                 angles='xy')
    ax[0].set_ylabel(r"$k_y [\frac{1}{\AA}]$")

    ax[1].quiver(kxlist, kylist, np.imag(sp[:, :, 0]), np.imag(sp[:, :, 1]),
               angles='xy')

    ax[1].set_xlabel(r"$k_x [\frac{1}{\AA}]$")
    ax[1].set_ylabel(r"$k_y [\frac{1}{\AA}]$")
    plt.show()    # fig = plt.figure()
