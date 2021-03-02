import os
import glob
import argparse
import numpy as np
import shutil
import importlib.util

from cued.plotting import read_dataset

######################################################################################################
# THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py #
######################################################################################################

#################################
# PARAMETERS OF THE TEST SCRIPT #
#################################
print_latex_pdf       = False
threshold_rel_error   = 0.1

def check_test(testdir):


    print('\n\n=====================================================\n\nStart with test:'
          '\n\n' + testdir + '\n\n')

    filename_params     = testdir + '/params.py'
    filename_run        = testdir + '/runscript.py'
    filename_pdf        = testdir + '/latex_pdf_files/CUED_summary.pdf'

    params = import_params(filename_params)

    print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

    if print_latex_pdf_really:
        os.system("echo '    save_latex_pdf = True' >> "+filename_params)

    assert os.path.isfile(filename_params),  'params.py is missing.'
    assert os.path.isfile(filename_run),     'runscript.py is missing.'

    # Reading in reference data
    time_data_ref, freq_data_ref, _dens_data_ref = read_dataset(testdir, prefix='reference_')
    assert time_data_ref is not None,  'Reference time_data is missing.'
    assert freq_data_ref is not None, 'Reference frequency_data is missing.'

    os.chdir(testdir)

    os.system('mpirun -np 2 python3 runscript.py')

    # Reading in generated data
    time_data, freq_data, _dens_data = read_dataset(os.getcwd())
    assert time_data is not None, '"time_data.dat" was not generated from the code'
    assert freq_data is not None, '"frequency_data.dat" was not generated from the code'

    # Load all relevant files and restrict data to max 10th order
    freq = freq_data['f/f0']

    # All indices between 0 and 10th order
    freq_idx = np.where(np.logical_and(0 <= freq, freq <= 10))[0]


    # Full emission
    I_E_dir_ref = freq_data_ref['I_E_dir'][freq_idx]
    I_ortho_ref = freq_data_ref['I_ortho'][freq_idx]
    I_E_dir = freq_data['I_E_dir'][freq_idx]
    I_ortho = freq_data['I_ortho'][freq_idx]
    print("\n\nMaxima of the emission spectra: ",
          "\nfull  E_dir: ", np.amax(np.abs(I_E_dir_ref)),
          "\nfull  ortho: ", np.amax(np.abs(I_ortho_ref)))
    check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, 'full')

    if params.split_current:
        # Intra + dtP emission
        I_intra_plus_dtP_E_dir_ref = freq_data_ref['I_intra_plus_dtP_E_dir'][freq_idx]
        I_intra_plus_dtP_ortho_ref = freq_data_ref['I_intra_plus_dtP_E_dir'][freq_idx]
        I_intra_plus_dtP_E_dir = freq_data['I_intra_plus_dtP_E_dir'][freq_idx]
        I_intra_plus_dtP_ortho = freq_data['I_intra_plus_dtP_E_dir'][freq_idx]
        print("\nintra plus dtP E_dir: ", np.amax(np.abs(I_intra_plus_dtP_E_dir_ref)),
            "\nintra plus dtP ortho: ", np.amax(np.abs(I_intra_plus_dtP_ortho_ref)))
        check_emission(I_intra_plus_dtP_E_dir, I_intra_plus_dtP_ortho,
                       I_intra_plus_dtP_E_dir_ref, I_intra_plus_dtP_ortho_ref,
                       'dtP')

    shutil.rmtree(testdir + '/__pycache__')
    for E0_dirname   in glob.glob(testdir + '/E0*'):   shutil.rmtree(E0_dirname)
    for PATH_dirname in glob.glob(testdir + '/PATH*'): shutil.rmtree(PATH_dirname)

    os.remove(testdir + '/time_data.dat')
    os.remove(testdir + '/frequency_data.dat')
    os.remove(testdir + '/params.txt')

    if print_latex_pdf_really:
        assert os.path.isfile(filename_pdf),  "The latex PDF is not there."
        os.system("sed -i '$ d' "+filename_params)
        filename_pdf_final  = testdir + '/CUED_summary_current_version.pdf'
        shutil.move(filename_pdf, filename_pdf_final)
        shutil.rmtree(testdir + '/latex_pdf_files')

    print('Test passed successfully.'
          '\n\n=====================================================\n\n')

    os.chdir("..")


def check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, name):
    relerror = (np.abs(I_E_dir + I_ortho) + 1.0E-90) / \
        (np.abs(I_E_dir_ref + I_ortho_ref) + 1.0E-90) - 1

    max_relerror = np.amax(np.abs(relerror))

    print("\n\nTesting the \"" + name + "\" emission spectrum I(omega):",
      "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", max_relerror,
        "\nThe threshold is:                                                                 ", threshold_rel_error, "\n")

    assert max_relerror < threshold_rel_error, "The \"" + name + "\" emission spectrum is not matching."


def import_params(filename_params):

    spec = importlib.util.spec_from_file_location("params", filename_params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    return params.params()


def check_params_for_print_latex_pdf(print_latex_pdf, params):

    if print_latex_pdf == True:
        if hasattr(params, 'save_latex_pdf'):
            print_latex_pdf_really = params.save_latex_pdf
        else:
            print_latex_pdf_really = True
    else:
        print_latex_pdf_really = False

    return print_latex_pdf_really


def main():

    dirpath = os.getcwd()

    print('\n\n=====================================================\n\n SBE CODE TESTER'
          '\n\n Executed in the directory:\n\n '+dirpath+
          '\n\n=====================================================\n\n')

    tests_path = dirpath + '/tests'
    count = 0

    for cdir in sorted(os.listdir(tests_path)):
        testdir = tests_path + '/' + cdir
        if os.path.isdir(testdir) and not cdir.startswith('norun'):
            print(testdir)
            count += 1
            check_test(testdir)

    assert count > 0, 'There are no test files with ending .reference in directory ' + tests_path

if __name__ == "__main__":
    main()
