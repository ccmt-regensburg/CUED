import matplotlib.pyplot as plt
import numpy as np

import sbe.plotting as splt

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 20

fullpath = "/loctmp/wij17778/29_sbe_code/sbe/scripts/runscripts/dist_0.00_Nk_in_path_10_num_paths_2/T1_1000_T2_1/chirp_0.000/phase_0.00/"

# Evaluation parameters for fast scanning
Iexactdata, Iapprox, Sol = splt.read_dataset(fullpath)

freqw = Iexactdata[3]
Int_exact_E_dir = Iexactdata[6]
Int_exact_ortho = Iexactdata[7]

Int_data_max = np.max(Int_exact_E_dir)

Int_exact_E_dir /= Int_data_max
Int_exact_ortho /= Int_data_max

if Iapprox is not None:
    Int_approx_E_dir = Iapprox[6]
    Int_approx_ortho = Iapprox[7]
    Int_approx_max = np.max(Int_approx_E_dir)
    Int_approx_E_dir /= Int_approx_max
    Int_approx_ortho /= Int_approx_max
    Int_exact_E_dir = np.vstack((Int_exact_E_dir, Int_approx_E_dir))
    Int_exact_ortho = np.vstack((Int_exact_ortho, Int_approx_ortho))


splt.fourier_dir_ortho(freqw, Int_exact_E_dir, Int_exact_ortho, xlim=(0, 21), ylim=(1e-24, 1),
                       savename='output.png')
