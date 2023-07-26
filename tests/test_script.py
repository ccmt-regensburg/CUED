import os
import glob
import argparse
import numpy as np
import shutil
import importlib.util
import importlib.machinery
import sys

from cued import HEADPATH
from cued.plotting import read_dataset
from cued.utility import ParamsParser

#########################################################################################################################################
# THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py --path tests --latex True --mpin 2 #
#########################################################################################################################################

#################################
# PARAMETERS OF THE TEST SCRIPT #
#################################
default_mpi_jobs = 1
print_latex_pdf = False
threshold_rel_error = 0.1
threshold_wrong_points = 0.01
default_tested_orders = 10

time_suffix = "time_data.dat"
freq_suffix = "frequency_data.dat"
params_suffix = "params.txt"
pdf_suffix = "latex_pdf_files"

def check_test(testdir, refdir):

    print('=====================================================\n'
                'Start with test:\n' + testdir + '\n against reference in \n' + refdir)

    filename_params     = testdir + '/params.py'
    filename_run        = testdir + '/runscript.py'

    params, current_mpi_num_procs, current_tested_orders = import_params(filename_params)

    # Get name of files (needed for MPI-tests)
    time_filenames = glob.glob1(refdir, "*" + time_suffix)
    refe_prefixes = [time_filename.replace(time_suffix, "") for time_filename in time_filenames]
    test_prefixes = [prefix.replace('reference_', '') for prefix in refe_prefixes]
    pdf_foldernames = [prefix + pdf_suffix for prefix in test_prefixes]

    if hasattr(params,"gabor_transformation") and params.gabor_transformation == True:
        gabor_filenames = glob.glob1(refdir, "reference_gabor" + "*" + freq_suffix)
        gabor_refe_prefixes = [freq_filename.replace(freq_suffix, "") for freq_filename in gabor_filenames]
        gabor_test_prefixes = [prefix.replace('reference_', '') for prefix in gabor_refe_prefixes]

    print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

    if print_latex_pdf_really:
        os.system("echo '    save_latex_pdf = True' >> " + filename_params)

    assert os.path.isfile(filename_params),  'params.py is missing.'
    assert os.path.isfile(filename_run),     'runscript.py is missing.'

    time_data_ref = []
    freq_data_ref = []

    for prefix in refe_prefixes:

        time_data_tmp, freq_data_tmp, _dens_data = read_dataset(refdir, prefix=prefix, mute=True)

        assert time_data_tmp is not None, 'Reference time_data is missing.'
        assert freq_data_tmp is not None, 'Reference frequency_data is missing.'

        time_data_ref.append(time_data_tmp)
        freq_data_ref.append(freq_data_tmp)

    if hasattr(params,"gabor_transformation") and params.gabor_transformation == True:
        gabor_freq_data_ref = []

        for prefix in gabor_refe_prefixes:
            _, freq_data_tmp, _ = read_dataset(refdir, prefix=prefix, mute=True)
            assert freq_data_tmp is not None, 'Reference frequency_data is missing.'
            gabor_freq_data_ref.append(freq_data_tmp)

    ##################################
    # Execute script in the testdir
    ##################################
    prev_dir = os.getcwd()
    os.chdir(testdir)
    os.system('mpirun -n ' + str(current_mpi_num_procs) + ' python -W ignore ' + testdir + '/runscript.py')
    os.chdir(prev_dir)
    ##################################

    # Reading in generated data
    for i, prefix in enumerate(test_prefixes):
        time_data, freq_data, _dens_data = read_dataset(testdir, prefix=prefix, mute=True)

        os.remove(testdir + '/' + prefix + params_suffix)
        os.remove(testdir + '/' + prefix + freq_suffix)
        os.remove(testdir + '/' + prefix + time_suffix)

        assert time_data is not None, '"time_data.dat" was not generated from the code'
        assert freq_data is not None, '"frequency_data.dat" was not generated from the code'

        freq_idx = build_comparable_data(i, freq_data, freq_data_ref, current_tested_orders)

        if hasattr(params, 'split_current'):
            if params.split_current:
                # Intra + dtP emission
                I_intra_plus_dtP_E_dir_ref = freq_data_ref[i]['I_intra_plus_dtP_E_dir'][freq_idx]
                I_intra_plus_dtP_ortho_ref = freq_data_ref[i]['I_intra_plus_dtP_ortho'][freq_idx]
                I_intra_plus_dtP_E_dir = freq_data['I_intra_plus_dtP_E_dir'][freq_idx]
                I_intra_plus_dtP_ortho = freq_data['I_intra_plus_dtP_ortho'][freq_idx]
                print("\nintra plus dtP E_dir: ", np.amax(np.abs(I_intra_plus_dtP_E_dir_ref)),
                      "\nintra plus dtP ortho: ", np.amax(np.abs(I_intra_plus_dtP_ortho_ref)))
                check_emission(I_intra_plus_dtP_E_dir, I_intra_plus_dtP_ortho,
                               I_intra_plus_dtP_E_dir_ref, I_intra_plus_dtP_ortho_ref,
                               'intra_plus_dtP')

        if hasattr(params, 'save_anom'):
            if params.save_anom:
                # Intra + anom emission
                I_anom_ortho_ref = freq_data_ref[i]['I_anom_ortho'][freq_idx]
                I_anom_ortho = freq_data['I_anom_ortho'][freq_idx]

                print("\nintra plus dtP E_dir: ", np.amax(np.abs(I_anom_ortho_ref)))
                check_emission(I_anom_ortho, I_anom_ortho, I_anom_ortho_ref, I_anom_ortho_ref, 'anom')


        if print_latex_pdf_really:
            foldername_pdf = testdir + '/' + prefix + pdf_suffix + '/'
            filename_pdf = foldername_pdf + 'CUED_summary.pdf'
            assert os.path.isfile(filename_pdf),  "The latex PDF is not there."
            os.system("sed -i '$ d' " + filename_params)
            os.rename(filename_pdf, testdir + '/' + prefix + 'CUED_summary.pdf')
            shutil.rmtree(foldername_pdf)

    if hasattr(params,"gabor_transformation") and params.gabor_transformation == True:
        # Reading in generated data from Gabor trafo
        for i, prefix in enumerate(gabor_test_prefixes):
            _, gabor_freq_data, _ = read_dataset(testdir, prefix=prefix, mute=True)

            os.remove(testdir + '/' + prefix + freq_suffix)

            assert gabor_freq_data is not None, f'"{prefix}_frequency_data.dat" was not generated from the code'

            build_comparable_data(i, gabor_freq_data, gabor_freq_data_ref, current_tested_orders)


    shutil.rmtree(testdir + '/__pycache__')
    for E0_dirname   in glob.glob(testdir + '/E0*'):   shutil.rmtree(E0_dirname)
    for PATH_dirname in glob.glob(testdir + '/PATH*'): shutil.rmtree(PATH_dirname)

    print('Test passed successfully.'
          '\n\n=====================================================\n\n')


def read_data(dir, prefix):

    # Reading in reference data
    time_data, freq_data, _dens_data = read_dataset(dir, prefix=prefix, mute=True)

    return time_data, freq_data


def build_comparable_data(freq_data_index, freq_data, freq_data_ref, num_tested_orders):
    # Load all relevant files and restrict data to max 10th order
    freq = freq_data['f/f0']
    print('Number of tested orders: ' + str(num_tested_orders))

    # All indices between 0 and 10th order
    freq_idx = np.where(np.logical_and(0 <= freq, freq <= num_tested_orders))[0]

    # Emission
    I_E_dir_ref = freq_data_ref[freq_data_index]['I_E_dir'][freq_idx]
    I_ortho_ref = freq_data_ref[freq_data_index]['I_ortho'][freq_idx]
    I_E_dir = freq_data['I_E_dir'][freq_idx]
    I_ortho = freq_data['I_ortho'][freq_idx]
    print("\n\nMaxima of the emission spectra: ",
        "\nfull  E_dir: ", np.amax(np.abs(I_E_dir_ref)),
        "\nfull  ortho: ", np.amax(np.abs(I_ortho_ref)))
    check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, 'full')
    return freq_idx


def check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, name):
    relerror = (np.abs(I_E_dir + I_ortho) + 1.0E-90) / \
               (np.abs(I_E_dir_ref + I_ortho_ref) + 1.0E-90) - 1

    error_list = [err for err in relerror if np.abs(err) >= threshold_rel_error]
    wrong_points = len(error_list)/len(relerror)

    max_relerror = np.amax(np.abs(relerror))

    print("\n\nTesting the \"" + name + "\" emission spectrum I(omega):",
          "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", max_relerror,
          "\nThe threshold is:                                                               ", threshold_rel_error, "\n"
          "\nThe relative deviation is above the threshold for " + str(wrong_points) + " percent of the spectrum \n"
          "\nThe threshold is:                                                               ", threshold_wrong_points, "\n")

    assert wrong_points < threshold_wrong_points, "The \"" + name + "\" emission spectrum is not matching."


def import_params(filename_params):
    """
    Imports the file dependent parameter file. If M/PI_NUM_PROCS is set
    it changes the number of started jobs for the file.
    """

    spec = importlib.util.spec_from_file_location("params", filename_params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    if hasattr(params, 'MPI_NUM_PROCS'):
        current_mpi_num_procs = params.MPI_NUM_PROCS
    else:
        if params.params().gauge == "velocity":
            if params.params().split_paths == False:
                num_ranks = params.params().Nk1 * params.params().Nk2
                os.system("echo '    parallelize_over_points = True' >> "+filename_params)
        else:
            num_ranks = params.params().Nk2

        if num_ranks < default_mpi_jobs:
            current_mpi_num_procs = num_ranks
        else:
            current_mpi_num_procs = default_mpi_jobs
    print(f"MG - Debug: Used {current_mpi_num_procs} ranks for {filename_params}")
    if hasattr(params, 'NUM_TESTED_ORDERS'):
        current_tested_orders = params.NUM_TESTED_ORDERS
    else:
        current_tested_orders = default_tested_orders

    return params.params(), current_mpi_num_procs, current_tested_orders

def check_params_for_print_latex_pdf(print_latex_pdf, params):

    # print_latex_pdf is the global variable
    if print_latex_pdf == True:
        if hasattr(params, 'save_latex_pdf'):
            print_latex_pdf_really = params.save_latex_pdf
        else:
            print_latex_pdf_really = print_latex_pdf
    else:
        print_latex_pdf_really = False

    return print_latex_pdf_really

def create_reference_data(testdir):
    print('=====================================================\n'
                'Create reference data in:\n' + testdir + '\n'
          '=====================================================\n')

    filename_params     = testdir + '/params.py'
    filename_run        = testdir + '/runscript.py'

    params, current_mpi_num_procs, current_tested_orders = import_params(filename_params)

    print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

    if print_latex_pdf_really:
        os.system("echo '    save_latex_pdf = True' >> "+filename_params)

    assert os.path.isfile(filename_params),  'params.py is missing.'
    assert os.path.isfile(filename_run),     'runscript.py is missing.'

    ##################################
    # Execute script in the testdir
    ##################################
    prev_dir = os.getcwd()
    os.chdir(testdir)
    os.system('mpirun -n ' + str(current_mpi_num_procs) + ' python -W ignore ' + testdir + '/runscript.py')
    for output_file in os.listdir(testdir):
        if not output_file.startswith('reference_') and ( output_file.endswith('.dat') or
                                                          output_file.endswith('.txt') ):
            os.rename(testdir + '/' + output_file, testdir + '/' + 'reference_' + output_file)
    os.chdir(prev_dir)
    ##################################

def tester(testpath, test_type):

    testpath = HEADPATH + '/' + testpath
    if (test_type == 'test'):
        print('=====================================================\n'
              'CUED CODE TESTER\n'
              'Executing tests in:\n' + testpath + '\n'
              '=====================================================')
    elif (test_type == 'reference'):
        print('=====================================================\n'
              'Create all referece data in:\n' + testpath + '\n'
              '=====================================================')
    count = 0

    for cdir in sorted(os.listdir(testpath)):
        testdir = testpath + '/' + cdir
        if os.path.isdir(testdir) and not cdir.startswith('norun') and not cdir.startswith('crosstest', 3):
            count += 1
            if (test_type == 'test'):
                check_test(testdir, testdir)
            elif (test_type == 'reference'):
                create_reference_data(testdir)
        if os.path.isdir(testdir) and cdir.startswith('crosstest', 3):
            for dummydir in sorted(os.listdir(testpath)):
                if dummydir.startswith(cdir[-2:]):
                    refdir = testpath + '/' + dummydir
            if (test_type == 'test'):
                try:
                    check_test(testdir, refdir)
                except UnboundLocalError:
                    raise FileNotFoundError(f"Directory with number {cdir[-2:]} for crosstest was not found in {testpath}")

    assert count > 0, 'There are tests in directory ' + testpath


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default='tests',
                        help="Relative testpath with respect to top level CUED dir.")
    parser.add_argument("-l", "--latex", type=bool, default=True,
                        help="Latex - PDF compilation.")
    parser.add_argument("-n", "--mpin", type=int, default=os.cpu_count()//2,
                        help="Number of mpi jobs")
    parser.add_argument("-t", "--test_type", type=str, default="test",
                        help="Do 'test' or redo 'reference' files.")
    args = parser.parse_args()

    return args.path, args.latex, args.mpin, args.test_type

if __name__ == "__main__":
    path, print_latex_pdf, default_mpi_jobs, test_type = parser()
    tester(path, test_type)
