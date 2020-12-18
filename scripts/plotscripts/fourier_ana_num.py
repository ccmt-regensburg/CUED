import matplotlib.pyplot as plt
import numpy as np

import sbe.plotting as splt

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 20

fullpath = "/loctmp/sea23537/Source/sbe/scripts/runscripts/ana/"
fullpath2 = "/loctmp/sea23537/Source/sbe/scripts/runscripts/num/"
# Evaluation parameters for fast scanning
Iexactdata, Iapprox, Sol = splt.read_dataset(fullpath)
Iexactdata2, Iapprox2, Sol2 = splt.read_dataset(fullpath2)

freqw = Iexactdata[3]
freqw2 = Iexactdata2[3]
Int_exact_E_dir = Iexactdata[6]
Int_exact_ortho = Iexactdata[7]
Int_exact_E_dir2 = Iexactdata2[6]
Int_exact_ortho2 = Iexactdata2[7]

# Int_data_max = np.max(Int_exact_E_dir)

# Int_exact_E_dir /= Int_data_max
# Int_exact_ortho /= Int_data_max

# if Iapprox is not None:
#     Int_approx_E_dir = Iapprox[6]
#     Int_approx_ortho = Iapprox[7]
#     Int_approx_max = np.max(Int_approx_E_dir)
#     Int_approx_E_dir /= Int_approx_max
#     Int_approx_ortho /= Int_approx_max
#     Int_exact_E_dir = np.vstack((Int_exact_E_dir, Int_approx_E_dir))
#     Int_exact_ortho = np.vstack((Int_exact_ortho, Int_approx_ortho))


splt.fourier_ana_num(freqw, Int_exact_E_dir, Int_exact_E_dir2, xlim=(0, 20), ylim=(1e-24, 1e-12),
                       )
