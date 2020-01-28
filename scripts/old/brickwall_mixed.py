"""
Brickwall lattice test to compare the approach of giving hopping matices to
the solver to giving a full k-space hamiltonian and its derivative.

"""
import math as ma
from mpl_toolkits.mplot3d import Axes3D # NOQA
import matplotlib.pyplot as plt
import numpy as np

from hfsbe import dipole
from hfsbe import fukui


def brickwall_hop():
    """
    Construct the hopping matrices of a brickwall lattice (graphene equivalent)

    """

    t = 1
    on = np.array([[0, t], [t, 0]])
    hopx = np.array([[0, 0], [t, 0]])
    hopx_d = hopx.T
    hopy = np.array([[0, 0], [t, 0]])
    hopy_d = hopy.T

    return  on, hopx, hopx_d, hopy, hopy_d

def brickwall_full(kx, ky):
    """
    Full k-space hamiltonian of a brickwall lattice

    """

    t = 1
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)

    return (t + ma.cos(kx) + ma.cos(ky))*sx + (ma.sin(kx) + ma.sin(ky))*sy

def brickwall_full_deriv(kx, ky):
    """
    Full k-space hamiltonian derivative of a brickwall lattice

    """

    t = 1
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)

    return -ma.sin(kx)*sx + ma.cos(kx)*sy, -ma.sin(ky)*sx + ma.cos(ky)*sy


if __name__ == "__main__":
    kxlist = np.linspace(-2*np.pi, 2*np.pi, 101)
    kylist = kxlist

    brickwall_hop = dipole.Dipole(brickwall_hop(), kxlist, kylist, gidx=0)
    brickwall_full = dipole.Dipole(brickwall_full, kxlist, kylist, gidx=0,
                                   hderivative=brickwall_full_deriv)

    fig = plt.figure()
    kxv, kyv = np.meshgrid(kxlist, kylist)
    ax0 = fig.add_subplot(211, projection='3d')
    ax0.plot_surface(kxv, kyv, brickwall_hop.e[:, :, 0])
    ax0.plot_surface(kxv, kyv, brickwall_hop.e[:, :, 1])
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.plot_surface(kxv, kyv, brickwall_full.e[:, :, 0])
    ax1.plot_surface(kxv, kyv, brickwall_full.e[:, :, 1])
    plt.show()

    sp = brickwall_hop.dipole_field(0, 1, energy_eps=10e-10)
    spf = brickwall_full.dipole_field(0, 1, energy_eps=10e-10)

    fig, ax = plt.subplots(2, 1)
    ax[0].quiver(kxlist, kylist, np.real(sp[:, :, 0]), np.real(sp[:, :, 1]),
                 angles='xy')

    ax[1].quiver(kxlist, kylist, np.real(spf[:, :, 0]), np.real(spf[:, :, 1]),
                 angles='xy')

    plt.xlabel(r"$k_y [\frac{1}{\AA}]$")
    plt.ylabel(r"$k_z [\frac{1}{\AA}]$")
    plt.show()
