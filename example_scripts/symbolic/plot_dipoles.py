import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 15]


def plot_dipoles(kxbz, kybz, Ax, Ay, title):
    # Plot real and imaginary part of all four fields 00, 01, 10, 11
    Axr = np.real(Ax)
    Axi = np.imag(Ax)

    Ayr = np.real(Ay)
    Ayi = np.imag(Ay)

    rnorm = np.sqrt(Axr**2 + Ayr**2)
    inorm = np.sqrt(Axi**2 + Ayi**2)

    Axnr = Axr/rnorm
    Aynr = Ayr/rnorm

    Axni = Axi/inorm
    Ayni = Ayi/inorm

    fig, ax = plt.subplots(2, 2)
    fig.subtitle(title, fontsize=16)

    valence = ax[0, 0].quiver(kxbz, kybz, Axnr[0, 0], Aynr[0, 0],
                              np.log(rnorm[0, 0]),
                              angles='xy', cmap='cool')
    ax[0, 0].set_title(r"$\Re(\vec{A}_{" + '-' + '-' + "})$")
    ax[0, 0].axis('equal')
    plt.colorbar(valence, ax=ax[0, 0])

    conduct = ax[0, 1].quiver(kxbz, kybz, Axnr[1, 1], Aynr[1, 1],
                              np.log(rnorm[1, 1]),
                              angles='xy', cmap='cool')
    ax[0, 1].set_title(r"$\Re(\vec{A}_{" + '+' + '+' + "})$")
    ax[0, 1].axis('equal')
    plt.colorbar(conduct, ax=ax[0, 1])

    dipreal = ax[1, 0].quiver(kxbz, kybz, Axnr[1, 0], Aynr[1, 0],
                              np.log(rnorm[1, 0]),
                              angles='xy', cmap='cool')
    ax[1, 0].set_title(r"$\Re(\vec{A}_{" + '+' + '-' + "})$")
    ax[1, 0].axis('equal')
    plt.colorbar(dipreal, ax=ax[1, 0])

    dipimag = ax[1, 1].quiver(kxbz, kybz, Axni[1, 0], Ayni[1, 0],
                              np.log(inorm[1, 0]),
                              angles='xy', cmap='cool')
    ax[1, 1].set_title(r"$\Im(\vec{A}_{" + '+' + '-' + "})$")
    ax[1, 1].axis('equal')
    plt.colorbar(dipimag, ax=ax[1, 1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
