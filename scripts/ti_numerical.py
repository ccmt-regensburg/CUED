import numpy as np
from hfsbe import dipole
from mpl_toolkits.mplot3d import Axes3D # NOQA
import matplotlib.pyplot as plt


def construct_slab(Z):

    t = 1
    so = np.array([[1, 0], [0, 1]], dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    m = 2.0

    g1 = np.kron(so, sz)
    g2 = np.kron(-sy, sx)
    g3 = np.kron(sx, sx)
    g4 = np.kron(-so, sy)

    on = m*g1
    hopx = -t*0.5*(g1 - 1j*g2)
    hopy = -t*0.5*(g1 - 1j*g3)
    hopz = -t*0.5*(g1 - 1j*g4)

    chain = np.eye(Z)
    chain_z = np.kron(chain, on)
    chain_z += np.kron(np.eye(Z, k=1), hopz)\
            + np.kron(np.eye(Z, k=-1), hopz.conj().T)
    chain_hop_x = np.kron(chain, hopx)
    chain_hop_x_d = chain_hop_x.conj().T
    chain_hop_y =  np.kron(chain, hopy)
    chain_hop_y_d = chain_hop_y.conj().T

    return chain_z, chain_hop_x, chain_hop_x_d, chain_hop_y, chain_hop_y_d

if __name__ == "__main__":
    h = construct_slab(10)
    kxlist = np.linspace(-np.pi, np.pi, 31)
    kylist = kxlist
    ti_dipole = dipole.Dipole(h, kxlist, kylist, gidx=4)

    fig = plt.figure()
    kxv, kyv = np.meshgrid(kxlist, kylist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kxv, kyv, ti_dipole.e[:, :, 19])
    ax.plot_surface(kxv, kyv, ti_dipole.e[:, :, 20])
    plt.xlabel(r"$k_x [\frac{1}{\AA}]$")
    plt.ylabel(r"$k_y [\frac{1}{\AA}]$")
    plt.show()

    sp = ti_dipole.dipole_field(19, 20, energy_eps=10e-10)

    fig, ax = plt.subplots(2, 1)
    ax[0].quiver(kxlist, kylist, np.real(sp[:, :, 0]), np.real(sp[:, :, 1]),
                 angles='xy')

    ax[1].quiver(kxlist, kylist, np.imag(sp[:, :, 0]), np.imag(sp[:, :, 1]),
                 angles='xy')
    
    plt.xlabel(r"$k_x [\frac{1}{\AA}]$")
    plt.ylabel(r"$k_y [\frac{1}{\AA}]$")
    plt.show()

