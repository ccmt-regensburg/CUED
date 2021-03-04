import os
import time
import numpy as np
from numpy.fft import *
import multiprocessing
from params import params

import cued.hamiltonian
from cued.utility import mkdir_chdir
from cued.main import sbe_solver, fourier_current_intensity, gaussian
from cued.plotting import read_dataset
from cued.utility import ConversionFactors as co

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = cued.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

    return dirac_system
def run(system):

    num_E_fields = params.num_E_fields
    dist_max     = params.dist_max
    weight       = dist_max / num_E_fields

    dists = np.linspace(weight/2, dist_max-weight/2, num_E_fields)

    E0list = params.E0*np.exp(-dists**2/2)

    ncpus = multiprocessing.cpu_count() - 1

    jobs = []

    for count, E0 in enumerate(E0list):

        params.E0 = E0

        print("\nSTARTED WITH E0 =", E0, "\n")

        mkdir_chdir("E0_"+str(E0))

        p = multiprocessing.Process(target=sbe_solver, args=(system, params, ))
        jobs.append(p)
        p.start()
        if (count+1) % ncpus == 0 or count+1 == len(E0list):
            for q in jobs:
                q.join()

        os.chdir('..')

    # collect all the currents from different E-field strenghts
    for count, E0 in enumerate(E0list):

        print("\nSTARTED WITH READING E0 =", E0, "\n")

        time_data, freq_data, _dens_data = read_dataset("./E0_"+str(E0))

        # integration weight
        scale_current = weight * dists[count]

        current_time_E_dir = time_data['j_E_dir'] * scale_current
        current_time_ortho = time_data['j_ortho'] * scale_current

        if count == 0:
            current_time_E_dir_integrated = np.zeros(np.size(current_time_E_dir), dtype=np.float64)
            current_time_ortho_integrated = np.zeros(np.size(current_time_ortho), dtype=np.float64)

        current_time_E_dir_integrated += current_time_E_dir.real
        current_time_ortho_integrated += current_time_ortho.real

    t = time_data['t']
    dt = params.dt*co.fs_to_au
    f = params.f*co.THz_to_au
    sigma = params.sigma*co.fs_to_au

    prefac_emission = 1/(3*(137.036**3))
    freq = fftshift(fftfreq(t.size, d=dt))

    Int_exact_E_dir, Int_exact_ortho, Iw_exact_E_dir, Iw_exact_ortho = fourier_current_intensity(
            current_time_E_dir_integrated, current_time_ortho_integrated,
            gaussian(t, sigma), dt, prefac_emission, freq)

    np.save('Iexact_integrated_over_E_fields', [t,
                           current_time_E_dir_integrated, current_time_ortho_integrated,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])

    # dummy test: safe the exact current as approximate current
    np.save('Iapprox_integrated_over_E_fields', [t,
                           current_time_E_dir_integrated, current_time_ortho_integrated,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])

if __name__ == "__main__":
    run(dirac())
