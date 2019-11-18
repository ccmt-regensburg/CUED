import numpy as np
from hfsbe import dipole
from mpl_toolkits.mplot3d import Axes3D # NOQA
import matplotlib.pyplot as plt


def construct_slab(W):

    t = 1
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    m = 0.0

    on = t*(2-m)*s3
    hopx = 1j*t/2*s1 - t/2*s3
    hopy = 1j*t/2*s2 - t/2*s3
    hopz = t/2*s3

    chain = np.eye(W)
    chain_x = np.kron(chain, on)
    chain_x += np.kron(np.eye(W, k=1), hopx)\
        + np.kron(np.eye(W, k=-1), hopx.conj().T)
    chain_hop_y = np.kron(chain, hopy)
    chain_hop_y_d = chain_hop_y.T.conj()
    chain_hop_z = np.kron(chain, hopz)
    chain_hop_z_d = chain_hop_z.T.conj()

    return chain_x, chain_hop_y, chain_hop_y_d, chain_hop_z, chain_hop_z_d


if __name__ == "__main__":
    h = construct_slab(2)
    kylist = np.linspace(-np.pi, np.pi, 31)
    kzlist = np.linspace(-np.pi, np.pi, 31)
    weyl_dipole = dipole.Dipole(h, kylist, kzlist, gauge_idx=2)

    fig = plt.figure()
    kyv, kzv = np.meshgrid(kylist, kzlist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kyv, kzv, weyl_dipole.e[:, :, 1])
    ax.plot_surface(kyv, kzv, weyl_dipole.e[:, :, 2])
    plt.show()
    sp = weyl_dipole.dipole_field(1, 2, energy_eps=0.5)
    
    # spf = weyl_fukui.dipole_field(9, 10)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].quiver(kylist, kzlist, np.real(sp[:, :, 0]), np.real(sp[:, :, 1]),
    #              angles='xy')

    # ax[1].quiver(kylist, kzlist, np.imag(sp[:, :, 0]), np.imag(sp[:, :, 1]),
    #              angles='xy')    breakpoint()
    plt.quiver(kylist, kzlist, np.imag(sp[:, :, 0]), np.imag(sp[:, :, 1]),
                 angles='xy')

    plt.xlabel(r"$k_y [\frac{1}{\AA}]$")
    plt.ylabel(r"$k_z [\frac{1}{\AA}]$")
    plt.show()
