import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import numpy as np

from sbe.plotting.colormap import whitedarkjet

plt.rcParams['text.usetex'] = True


def cep_plot(freqw, phaselist, intensity,
             xlim=(0, 30), inorm=None,
             xlabel=r'Frequency $\text{ in } \omega/\omega_0$',
             ylabel=r'phase $\phi$',
             yticks=None,
             supertitle=None, title=None, savename=None):

    freqw = freqw.real
    intensity = intensity.real

    if inorm is not None:
        intensity /= inorm.real

    # Limit data to visible window
    lorder, rorder = xlim
    lidx = np.where(freqw[0, :] > lorder)[0][0]
    ridx = np.where(freqw[0, :] < rorder)[0][-1] + 1

    freqw = freqw[:, lidx:ridx]
    intensity = intensity[:, lidx:ridx]

    imax = np.max(intensity)
    imin = np.min(intensity)
    imax_log = np.log10(imax)

    imin_log = np.log10(imin)
    F, P = np.meshgrid(freqw[0], phaselist)

    _fig, ax = plt.subplots()
    logspace = np.logspace(imin_log, imax_log, 1000)
    cont = ax.contourf(F, P, intensity, levels=logspace,
                       locator=ticker.LogLocator(),
                       cmap=whitedarkjet,
                       norm=colors.LogNorm(vmin=imin, vmax=imax))

    int_imin_log = int(imin_log)
    int_imax_log = int(imax_log)
    tickposition = np.logspace(int_imin_log, int_imax_log,
                               num=np.abs(int_imax_log - int_imin_log) + 1)

    cb = plt.colorbar(cont, ticks=tickposition)
    if inorm is not None:
        normlabel = r'I_\mathrm{hh}^\mathrm{max}'
        cb.set_label(r'$I_\mathrm{hh}/' + normlabel + '$')
        cb.ax.set_title(r'$' + normlabel + ' = {:.2e}'.format(inorm)
                        + r'\si{[a.u.]}$')
    else:
        cb.set_label(r'$I_\mathrm{hh}$')

    ax.set_xticks(np.arange(xlim[1] + 1))

    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    ax.grid(True, axis='x', ls='--')
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if supertitle is not None:
        plt.suptitle(supertitle)

    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
