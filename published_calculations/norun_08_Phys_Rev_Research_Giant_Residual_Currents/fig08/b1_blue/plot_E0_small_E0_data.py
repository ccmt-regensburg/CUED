from params_small_E0_data import params
from cued.utility import ConversionFactors as CoFa
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_E0(sweepList, listCurrent, savename=None, tikz=False, logx=False, logy=False, loga=False):

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    plt.plot(sweepList, listCurrent, 'x')
    if loga or logy:
        plt.plot(sweepList, -listCurrent, 'o', color='C0', mfc='none' )
    plt.grid()

    if logx:
    
        plt.xscale('log')
        savename = 'lnx_' + savename
    if logy:
        plt.yscale('log')
        savename = 'lny_' + savename
    if loga:
        plt.xscale('log')
        plt.yscale('log')
        savename = 'ln_' + savename

    if not tikz:
        plt.title(f"Current over Electric Field")
    plt.xlabel("E0 in MV/cm")
    plt.ylabel("DC remnant in A")

    plt.legend()

    if savename == None and tikz == False:
        plt.show()
    elif tikz:
        import tikzplotlib as tpl

        tpl.save(savename + '.tex')
        plt.close()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    else:
        plt.savefig(savename + '.pdf')
        plt.show()

def run():

    listE0 = params.E0

    res_current_string = "residual_currents_small_E0_data"

    if not res_current_string in os.listdir(os.curdir):
    
            remnant_list = []
            for files in sorted(os.listdir(os.curdir)):
                if files.startswith('Nk2=360'):
                    if files.endswith('time_data.dat'):
                        path = os.getcwd()
                        time_data = np.genfromtxt(path + '/' + files, names=True, encoding='utf8', deletechars='')
                        remnant_list.append(time_data['j_E_dir'][-1])

            np.savez(res_current_string, E0=listE0, j_DC=remnant_list)

    savename = "DC_remnant_E0_sweep_div_small_E0_data"
    lists = np.load(res_current_string + '.npz')
    E0_list = lists['E0']
    remnant_list = lists['j_DC']

    # automatically use all available data
    count = remnant_list.size
    # manual
    # count = 40

    plot_E0(E0_list[:count], remnant_list[:count], savename=savename, loga=True, tikz=False)
    plot_E0(E0_list[:count], remnant_list[:count], savename=savename, loga=True, tikz=True)

if __name__ == "__main__":

    run()
    
