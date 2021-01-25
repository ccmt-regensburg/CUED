import matplotlib.pyplot as plt
import numpy as np

import sbe.plotting as splt

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 28


# Evaluation parameters for fast scanning
fullpath = #INSERT TEST PATH
Iexactdata, Iapprox, Sol = splt.read_datasets(fullpath)

freqw = Iexactdata[:, 3]
Int_exact_E_dir = Iexactdata[:, 6]
Int_exact_ortho = Iexactdata[:, 7]

phaselist = np.linspace(0, np.pi, 20)
Int_data = Int_exact_E_dir + Int_exact_ortho

Int_data_max = np.max(Int_data).real

subtitle = title['H'] + ' ' + title['E'] + ' ' + title['w']
title = title['fwhm'] + ' ' + title['chirp'] + ' '+ title['T1'] + ' ' \
    + title['T2']
splt.cep_plot(freqw, phaselist, Int_data, xlim=(13, 21), inorm=Int_data_max,
              subtitle=None, title=None, savename=savename + '.png')
