import matplotlib.pyplot as plt
import numpy as np

from hfsbe.utility import conversion_factors as co


def fourier_total(freqw, data_dir, data_ortho,
                  xlim=(0, 30), ylim=(10e-15, 1),
                  xlabel=r'Frequency $\text{ in } \omega/\omega_0$',
                  ylabel=r'Intensity $\text{ in atomic units}$',
                  paramlegend=None, suptitle=None, savename=None):
    """
    Plots parallel and orthogonal data
    """
    fig, ax = plt.subplots(1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[1]+1))
    ax.grid(True, axis='x', ls='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    data_total = data_dir + data_ortho
    for freq, data in zip(freqw, data_total):
        ax.semilogy(freq, data/np.max(data))

    if (suptitle is not None):
        fig.suptitle(suptitle)

    if (paramlegend is not None):
        ax.legend(paramlegend)

    if (savename is None):
        plt.show()
    else:
        plt.savefig(savename)


def fourier_dir_ortho(freqw, data_dir, data_ortho, xlim=(0.2, 30),
                      ylim=(10e-15, 1),
                      xlabel=r'Frequency $\omega/\omega_0$',
                      ylabel=r'Intensity $\text{ in atomic units}$',
                      paramlegend=None, suptitle=None,
                      savename=None):
    fig, ax = plt.subplots(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[1]+1))
    ax.grid(True, axis='x', ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.semilogy(freqw, data_dir.T)
    ax.semilogy(freqw, data_ortho.T)

    if (suptitle is not None):
        fig.suptitle(suptitle)

    if (paramlegend is not None):
        ax.legend(paramlegend)

    if (savename is None):
        plt.show()
    else:
        plt.savefig(savename)


def fourier_dir_ortho_split(freqw, data_dir, data_ortho, xlim=(0.2, 30),
                            ylim=(10e-15, 1),
                            xlabel=r'Frequency $\omega/\omega_0$',
                            ylabel=r'Intensity $\text{ in atomic units}$',
                            paramlegend=None, suptitle=None,
                            savename=None):
    fig, ax = plt.subplots(2)
    for a in ax:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.set_xticks(np.arange(31))
        a.grid(True, axis='x', ls='--')
        a.set_ylabel(ylabel)
    ax[0].set_title(r'$\mathbf{E}$ parallel')
    ax[1].set_title(r'$\mathbf{E}$ orthogonal')
    ax[1].set_xlabel(xlabel)
    for freq, data_d, data_o in zip(freqw, data_dir, data_ortho):
        ax[0].semilogy(freq, data_d)
        ax[1].semilogy(freq, data_o)

    if (suptitle is not None):
        fig.suptitle(suptitle)

    if (paramlegend is not None):
        ax[0].legend(paramlegend)

    if (savename is None):
        plt.show()
    else:
        plt.savefig(savename)
