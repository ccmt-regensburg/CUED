"""
Graphene dipole field test.

"""
from mpl_toolkits.mplot3d import Axes3D # NOQA
import math as ma
import matplotlib.pyplot as plt
import numpy as np

from hfsbe import dipole


def hamiltonian(kx, ky):
    """
    Construct a full k-space Graphene Hamiltonian.

    """

    t = -1
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)

    arg1 = kx
    arg2 = -kx/2 + ma.sqrt(3)*ky/2
    arg3 = -kx/2 - ma.sqrt(3)*ky/2

    H = (ma.cos(arg1) + ma.cos(arg2) + ma.cos(arg3))*sx -\
        (ma.sin(arg1) + ma.sin(arg2) + ma.sin(arg3))*sy
    H *= -t

    return H


def hderivative(kx, ky):
    """
    Derivative of full k-space Graphene Hamiltonian

    """

    t = -1
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)

    pre = ma.sqrt(3)/2
    arg1 = kx
    arg2 = -kx/2 + pre*ky
    arg3 = -kx/2 - pre*ky

    dxH = (-ma.sin(arg1) + 0.5*ma.sin(arg2) + 0.5*ma.sin(arg3))*sx -\
          (ma.cos(arg1) - 0.5*ma.cos(arg2) - 0.5*ma.cos(arg3))*sy
    dxH *= -t

    dyH = (-pre*ma.sin(arg2) + pre*ma.sin(arg3))*sx -\
          (pre*ma.cos(arg2) - pre*ma.cos(arg3))*sy
    dyH *= -t

    return dxH, dyH


if __name__ == "__main__":

    # Same spacing for kx and ky used here, important for derivatives
    kxlist = np.linspace(0, np.pi, 31)
    kylist = kxlist

    graphene_dipole = dipole.Dipole(hamiltonian, kxlist, kylist,
                                    hderivative=hderivative, gidx=1)

    fig = plt.figure()
    kxv, kyv = np.meshgrid(kxlist, kylist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kxv, kyv, graphene_dipole.e[:, :, 0])
    ax.plot_surface(kxv, kyv, graphene_dipole.e[:, :, 1])
    plt.show()

    sp = graphene_dipole.dipole_field(0, 1, energy_eps=0.5)
    breakpoint()
    # plt.quiver(kxlist, kylist, np.real(sp[:, :, 0]), np.real(sp[:, :, 1]),
    #              angles='xy')
    plt.quiver(kxlist, kylist, np.imag(sp[:, :, 0]), np.imag(sp[:, :, 1]),
                 angles='xy')
    plt.xlabel(r"$k_x [\frac{1}{\AA}]$")
    plt.ylabel(r"$k_y [\frac{1}{\AA}]$")
    plt.show()
