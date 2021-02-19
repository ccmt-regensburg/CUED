import os
import numpy as np
from numpy.fft import fftshift, fftfreq
import multiprocessing
import cued.hamiltonian
import cued.dipole
from cued.utility import ConversionFactors as co
from cued.utility import mkdir_chdir
from cued.main import sbe_solver, fourier_current_intensity, gaussian
from cued.plotting import read_dataset

from params import params

def dirac():
    A = 0.1974      # Fermi velocity

    dirac_system = cued.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

    return dirac_system

def run(system):

    ncpus = min( multiprocessing.cpu_count() - 1, params.Nk2)

    jobs = []

    print("\nStart parallel execution on", ncpus, "CPUs.\n")

    for Nk2_idx_ext in range(params.Nk2):

        params.Nk2_idx_ext = Nk2_idx_ext

        print("\nSTARTED WITH PATH =", Nk2_idx_ext + 1, "\n")

        mkdir_chdir("PATH_"+str(Nk2_idx_ext+1))

        p = multiprocessing.Process(target=sbe_solver, args=(system, params, ))
        jobs.append(p)
        p.start()
        if (Nk2_idx_ext+1) % ncpus == 0 or Nk2_idx_ext+1 == params.Nk2:
            for q in jobs:
                q.join()

        os.chdir('..')

    # collect all the currents from different k-paths
    for Nk2_idx_ext in range(params.Nk2):

        print("\nSTARTED WITH READING PATH =", Nk2_idx_ext + 1, "\n")

        Iexactdata, Iapproxdata, Sol = read_dataset("./PATH_"+str(Nk2_idx_ext+1)+"/")

        current_time_E_dir = Iexactdata[1]
        current_time_ortho = Iexactdata[2]

        if Nk2_idx_ext == 0:
            current_time_E_dir_integrated = np.zeros(np.size(current_time_E_dir), dtype=np.float64)
            current_time_ortho_integrated = np.zeros(np.size(current_time_ortho), dtype=np.float64)

        current_time_E_dir_integrated += current_time_E_dir.real
        current_time_ortho_integrated += current_time_ortho.real

    t = Iexactdata[0]
    dt = params.dt*co.fs_to_au
    w = params.w*co.THz_to_au
    alpha = params.alpha*co.fs_to_au

    prefac_emission = 1/(3*(137.036**3))
    freq = fftshift(fftfreq(t.size, d=dt))

    Int_exact_E_dir, Int_exact_ortho, Iw_exact_E_dir, Iw_exact_ortho = fourier_current_intensity(
            current_time_E_dir_integrated, current_time_ortho_integrated,
            gaussian(t, alpha), dt, prefac_emission, freq)

    np.save('Iexact_fully_k_integrated', [t,
                           current_time_E_dir_integrated, current_time_ortho_integrated,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])

    # dummy
    np.save('Iapprox_fully_k_integrated', [t,
                           current_time_E_dir_integrated, current_time_ortho_integrated,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])

if __name__ == "__main__":
    run(dirac())
