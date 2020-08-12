import numpy as np
import os


def read_datasets(subpaths):
    """
    Specify a path and read all subfolders
    read_dataset defines the individual order
    """
    Soldata_container = []
    Iexactdata_container = []
    Iapproxdata_container = []

    for i, subpath in enumerate(subpaths):
        print("Evaluating " + subpath + " data", end='\n\n')
        Iexactdata, Iapproxdata, Soldata = read_dataset(subpath)
        Iexactdata_container.append(Iexactdata)
        Iapproxdata_container.append(Iapproxdata)
        Soldata_container.append(Soldata)

    return np.array(Iexactdata_container), np.array(Iapproxdata_container),\
        np.array(Soldata_container)


def read_dataset(path):
    """
    Read the data from a specific folder;
    Emission exact
    [t, I_exact_E_dir, I_exact_ortho, freq/w, Iw_exact_E_dir, Iw_exact_ortho,
     Int_exact_E_dir, Int_exact_ortho]
    Emission approx
    [t, I_E_dir, I_ortho, freq/w, Iw_E_dir, Iw_ortho, Int_E_dir, Int_ortho]
    Solution
    [t, Solution, paths, electric_field, A_field]

    Parameters
    ----------
    path : String
        Path to the dataset

    Returns
    -------
    Iexactdata : np.ndarray
        Exact current and emission data
    Iapprox : np.ndarray
        Approx. current and emission data (kira & koch formula)
    Solution : np.ndarray
        Full solution density, paths, electric field and vector potential
    """
    filelist = os.listdir(path)
    Soldata = None
    Iexactdata = None
    Iapproxdata = None

    for filename in filelist:
        # Emissions I
        # [t, solution, paths, electric_field, A_field]
        if ('Sol_' in filename):
            print("Reading :", path, filename)
            Soldict = np.load(path + filename)
            Soldata = np.array([Soldict['t'], Soldict['solution'],
                                Soldict['paths'], Soldict['electric_field'],
                                Soldict['A_field']])

        # Emissions Iexact
        # [t, I_exact_E_dir, I_exact_ortho, freq/w, Iw_exact_E_dir,
        # Iw_exact_ortho, Int_exact_E_dir, Int_exact_ortho]
        if ('Iexact_' in filename):
            print("Reading :", path, filename)
            Iexactdata = np.array(np.load(path + filename))

        # Emission approximate
        if ('Iapprox_' in filename):
            print("Reading :", path, filename)
            Iapproxdata = np.array(np.load(path + filename))

    return Iexactdata, Iapproxdata, Soldata
