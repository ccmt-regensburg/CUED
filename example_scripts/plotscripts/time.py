import matplotlib.pyplot as plt
import numpy as np

import sbe.plotting as splt

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 20


fullpath = #INSERT TEST PATH
Iexactdata, Iapprox, Sol = splt.read_dataset(fullpath)
time = Iexactdata[0]

J_exact_E_dir = Iexactdata[1]
J_exact_ortho = Iexactdata[2]

if Iapprox is not None:
    J_approx_E_dir = Iapprox[1]
    J_approx_ortho = Iapprox[2]
    J_exact_E_dir = np.vstack((J_exact_E_dir, J_approx_E_dir))
    J_exact_ortho = np.vstack((J_exact_ortho, J_approx_ortho))

subtitle = title['H'] + ' ' + title['E'] + ' ' + \
    title['w'] + ' ' + dist.replace('_', '=')
title = title['fwhm'] + ' ' + title['chirp'] + ' ' + \
    title['T1'] + ' ' + title['T2'] + ' ' + title['phase']
splt.time_dir_ortho(time, J_exact_E_dir, J_exact_ortho,
                    subtitle=subtitle, title=title,
                    savename=None, marker='.')
