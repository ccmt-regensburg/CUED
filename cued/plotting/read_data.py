import numpy as np
import os

def read_datasets(subpaths):
    """
    Specify a path and read all subfolders
    read_dataset defines the individual order
    """
    time_data_container = []
    freq_data_container = []
    dens_data_container = []

    for subpath in subpaths:
        print("Evaluating " + subpath + " data", end='\n\n')
        time_data, freq_data = read_dataset(subpath)
        time_data_container.append(time_data)
        freq_data_container.append(freq_data)
        dens_data_container.append(dens_data)

    return time_data_container, freq_data_container, dens_data_container

def read_dataset(path, prefix='', suffix=''):
    """
    Read the data from a specific folder;
    Create memory views of all datasets

    Parameters
    ----------
    path : String
        Path to the dataset

    Returns
    -------
    time_data : np.ndarray
        time dependent data
    freq_data : np.ndarray
        fourier/frequency data
    dens_data : np.ndarray
        data from the full density matrix
    """
    filelist = os.listdir(path)
    time_data = None
    freq_data = None
    dens_data = None

    time_string = prefix + 'time_data' + suffix
    freq_string = prefix + 'frequency_data' + suffix
    dens_string = prefix + 'density_data' + suffix

    for filename in filelist:
        # Time data
        filepath = path + '/' + filename
        if (filename.startswith(time_string) and '.dat' in filename):
            print("Reading time data:", filepath)
            time_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

        # Frequency data
        if (filename.startswith(freq_string) and '.dat' in filename):
            print("Reading frequency data:", filepath)
            freq_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

        # Density data
        if (filename.startswith(dens_string) and '.dat' in filename):
            print("Reading density:", filepath)
            dens_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

    return time_data, freq_data, dens_data
