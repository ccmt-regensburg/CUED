import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cued.plotting import read_dataset, fourier_dir_ortho_angle, fourier_dir_ortho_angle_polar

time, freq, _dens = read_dataset('./test', prefix='reference_s90_')

fourier_dir_ortho_angle(freq['f/f0'], freq['Re[j_E_dir]'], freq['Im[j_E_dir]'],
                        freq['Re[j_ortho]'], freq['Im[j_ortho]'],
                        savename='lol', xlim=(15, 20))

fourier_dir_ortho_angle_polar(freq['f/f0'], freq['Re[j_E_dir]'], freq['Im[j_E_dir]'],
                              freq['Re[j_ortho]'], freq['Im[j_ortho]'], savename='polar')
