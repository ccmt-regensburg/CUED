"""
Qi-Wu-Zhang Chern Insulator implementation

"""
import math as ma
from mpl_toolkits.mplot3d import Axes3D # NOQA
import matplotlib.pyplot as plt
import numpy as np

from hfsbe import dipole
from hfsbe import fukui


def qwz(kx, ky):
    """
    Full k-space hamiltonian of a QWZ lattice

    """

    global m
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex)

    return ma.sin(kx)*sx + ma.sin(ky)*sy + [m-ma.cos(kx)-ma.cos(ky)]*sz

def qwz_deriv(kx, ky):
    """
    Full k-space hamiltonian derivative of a QWZ lattice

    """

    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex)

    return ma.cos(kx)*sx + ma.sin(kx)*sz, ma.cos(ky)*sy + ma.sin(ky)*sz


if __name__ == "__main__":
    kxlist = np.linspace(-np.pi, np.pi, 21)
    kylist = kxlist

    qwzm = []

    m = 1.5
    qwzm.append(dipole.Dipole(qwz, kxlist, kylist, gauge_idx=0, hderivative=qwz_deriv))
    
    m = 2.0
    qwzm.append(dipole.Dipole(qwz, kxlist, kylist, gauge_idx=0, hderivative=qwz_deriv))

    m = 2.5
    qwzm.append(dipole.Dipole(qwz, kxlist, kylist, gauge_idx=0, hderivative=qwz_deriv))
    
    fig = plt.figure()
    kxv, kyv = np.meshgrid(kxlist, kylist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kxv, kyv, qwzm[1].e[:, :, 0])
    ax.plot_surface(kxv, kyv, qwzm[1].e[:, :, 1])
    plt.show()

    fields = []
    fields.append(qwzm[0].dipole_field(0, 1, energy_eps=5e-1))
    fields.append(qwzm[1].dipole_field(0, 1, energy_eps=5e-1))
    fields.append(qwzm[2].dipole_field(0, 1, energy_eps=5e-1))

    fig, ax = plt.subplots(2, 3)
    ax[0][0].set_title(r"$m=1.5$")
    ax[0][0].quiver(kxlist, kylist, np.real(fields[0][:, :, 0]), np.real(fields[0][:, :, 1]),
                    angles='xy')
    ax[1][0].quiver(kxlist, kylist, np.imag(fields[0][:, :, 0]), np.imag(fields[0][:, :, 1]),
                    angles='xy')
    ax[0][1].set_title(r"$m=2.0$")
    ax[0][1].quiver(kxlist, kylist, np.real(fields[1][:, :, 0]), np.real(fields[1][:, :, 1]),
                    angles='xy')
    ax[1][1].quiver(kxlist, kylist, np.imag(fields[1][:, :, 0]), np.imag(fields[1][:, :, 1]),
                    angles='xy')
    ax[0][2].set_title(r"$m=2.5$")
    ax[0][2].quiver(kxlist, kylist, np.real(fields[2][:, :, 0]), np.real(fields[2][:, :, 1]),
                    angles='xy')
    ax[1][2].quiver(kxlist, kylist, np.imag(fields[2][:, :, 0]), np.imag(fields[2][:, :, 1]),
                    angles='xy')
    ax[0][0].set_xlabel(r"$k_y [\frac{1}{\AA}]$")
    ax[0][0].set_ylabel(r"$k_z [\frac{1}{\AA}]$")
    plt.show()
