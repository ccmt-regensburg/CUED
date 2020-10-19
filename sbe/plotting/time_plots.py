import matplotlib.pyplot as plt
import numpy as np

from sbe.utility import conversion_factors as co


def time_grid(time, kpath, electric_field, current, band_structure,
              density_center, standard_deviation, e_fermi=0.2,
              electric_field_legend=None, current_legend=None,
              band_structure_legend=None, density_center_legend=None,
              standard_deviation_legend=None, timelim=None, energylim=None,
              bzboundary=None, savename=None, si_units=True):

    if (si_units):
        time *= co.au_to_fs
        kpath /= co.au_to_as
        electric_field *= co.au_to_MVpcm
        current *= co.au_to_Amp
        band_structure *= co.au_to_eV
        density_center /= co.au_to_as
        standard_deviation /= co.au_to_as
    else:
        e_fermi *= co.eV_to_au

    ########################################
    # Electric field
    ########################################
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    ax1.plot(time, electric_field.T)
    if (si_units):
        ax1.set_xlabel(r'$t \text{ in } \si{fs}$')
        ax1.set_ylabel(r'$E \text{ in } \si{MV/cm}$')
    else:
        ax1.set_xlabel(r'$t \text{ in atomic units}$')
        ax1.set_ylabel(r'$E \text{ in atomic units}$')

    ax1.set_title(r'Electric Field')
    ax1.grid(which='major', axis='x', linestyle='--')
    if (electric_field_legend is not None):
        ax1.legend(electric_field_legend)

    ########################################
    # Current
    ########################################
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    ax2.plot(time, current.T)
    if (si_units):
        ax2.set_xlabel(r'$t \text{ in } \si{fs}$')
        ax2.set_ylabel(r'$j \text{ in } \si{A}$')
    else:
        ax2.set_xlabel(r'$t \text{ in atomic units}$')
        ax2.set_ylabel(r'$j \text{ in atomic units}$')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.set_title(r'Current Density')
    ax2.grid(which='major', axis='x', linestyle='--')
    ax2.axhline(y=0, linestyle='--', color='grey')
    if (current_legend is not None):
        ax2.legend(current_legend)

    ########################################
    # Band structure
    ########################################
    kpath_min = np.min(density_center)
    kpath_max = np.max(density_center)
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    # Number of band structures to plot
    band_num = np.size(band_structure, axis=0)
    ax3.plot(band_structure.T, np.tile(kpath, (band_num, 1)).T)
    if (si_units):
        ax3.set_xlabel(r'$\epsilon \text{ in } \si{eV}$')
        ax3.set_ylabel(r'$k \text{ in } \si{1/\angstrom}$')
    else:
        ax3.set_xlabel(r'$\epsilon \text{ in atomic units}$')
        ax3.set_ylabel(r'$k \text{ in atomic units}$')
    if (energylim is not None):
        ax3.set_xlim(energylim)
    ax3.axvline(x=e_fermi, linestyle=':', color='black')
    ax3.set_ylim(-kpath_max*1.05, kpath_max*1.05)
    ax3.axhline(y=kpath_min, linestyle='--', color='grey')
    ax3.axhline(y=0, linestyle='--', color='grey')
    ax3.axhline(y=kpath_max, linestyle='--', color='grey')

    ax3.set_title(r'Band Structure')
    if (band_structure_legend is not None):
        ax3.legend(band_structure_legend)

    ########################################
    # Density data
    ########################################
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2, sharey=ax3)
    ax4.plot(time, density_center[:-1].T)
    ax4.plot(time, density_center[-1], linestyle=':', color='red')
    if (si_units):
        ax4.set_xlabel(r'$t \text{ in } \si{fs}$')
    else:
        ax4.set_xlabel(r'$t \text{ in atomic units}$')
    ax4.set_title(r'Density Center of Mass')
    ax4.grid(which='major', axis='x', linestyle='--')
    ax4.axhline(y=kpath_min, linestyle='--', color='grey')
    ax4.axhline(y=0, linestyle='--', color='grey')
    ax4.axhline(y=kpath_max, linestyle='--', color='grey')
    plt.setp(ax4.get_yticklabels(), visible=False)
    if (density_center_legend is not None):
        ax4.legend(density_center_legend)

    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax5.set_title(r'Density Standard Deviation')
    ax5.plot(time, standard_deviation.T)
    ax5.yaxis.set_label_position("right")
    ax5.yaxis.tick_right()
    if (si_units):
        ax5.set_xlabel(r'$t \text{ in } \si{fs}$')
        ax5.set_ylabel(r'$\sigma \text{ in } \si{1/\angstrom}$')
    else:
        ax5.set_xlabel(r'$t \text{ in atomic units}$')
        ax5.set_ylabel(r'$\sigma \text{ in atomic units}$')

    if (standard_deviation_legend is not None):
        ax5.legend(standard_deviation_legend)

    if (timelim is not None):
        ax1.set_xlim(timelim)
        ax2.set_xlim(timelim)
        ax4.set_xlim(timelim)
        ax5.set_xlim(timelim)

    if (bzboundary is not None):
        ax3.set_title(r'Band Structure $k_\mathrm{BZ}='
                      + '{:.3f}'.format(bzboundary) + r'[\si{1/\angstrom}]$')
        ax3.axhline(y=bzboundary, linestyle=':', color='green')
        ax3.axhline(y=-bzboundary, linestyle=':', color='green')
        ax4.axhline(y=bzboundary, linestyle=':', color='green')
        ax4.axhline(y=-bzboundary, linestyle=':', color='green')

    plt.tight_layout()
    if (savename is not None):
        plt.savefig(savename)
    else:
        plt.show()


def time_dir_ortho_angle(time, current_dir, current_ortho, current_legend=None,
                         savename=None, si_units=True):

    if si_units:
        time *= co.au_to_fs
        current_dir *= co.au_to_Amp
        current_ortho *= co.au_to_Amp

    time = time.real
    current_dir = current_dir.real
    current_ortho = current_ortho.real

    _fig, ax = plt.subplots(2)
    ax[0].plot(time, current_dir.T, marker='.')
    ax[0].plot(time, current_ortho.T, linestyle='--')

    angle_data = np.arctan(current_ortho/current_dir)
    ax[1].plot(time, angle_data.T)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)


def time_dir_ortho(time, current_dir, current_ortho, xlim=None, ylim=None,
                   xlabel=r'Time in atomic units', ylabel=r'Current in atomic units',
                   marker=None, paramlegend=None, supertitle=None, title=None, savename=None,
                   si_units=True):

    time = time.real
    current_dir = current_dir.real
    current_ortho = current_ortho.real

    if si_units:
        time *= co.au_to_fs
        current_dir *= co.au_to_Amp*1e5
        current_ortho *= co.au_to_Amp*1e5
        xlabel = r'Time in fs'
        ylabel = r'Current in $\mu$A'

    _fig, ax = plt.subplots(1)
    _lines_dir = ax.plot(time.T, current_dir.T, marker=marker)
    plt.gca().set_prop_cycle(None)
    _lines_ortho = ax.plot(time.T, current_ortho.T, linestyle='--', marker=marker)

    ax.grid(True, axis='x', ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if paramlegend is not None:
        ax.legend(paramlegend)

    if supertitle is not None:
        plt.suptitle(supertitle)

    if title is not None:
        ax.set_title(title)

    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
