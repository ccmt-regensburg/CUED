import os
import glob
import argparse
import numpy as np
import shutil
import cued.plotting as splt
import importlib.util

######################################################################################################
# THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py #
######################################################################################################

def check_test(testdir):

    #################################
    # PARAMETERS OF THE TEST SCRIPT #
    #################################
    print_latex_pdf       = False
    threshold_rel_error   = 0.1

    print('\n\n=====================================================\n\nStart with test:'
          '\n\n' + testdir + '\n\n')

    filename_params     = testdir + '/params.py'
    filename_run        = testdir + '/runscript.py'
    filename_pdf        = testdir + '/latex_pdf_files/CUED_summary.pdf'

    params = import_params(filename_params)

    print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

    if print_latex_pdf_really:
        os.system("echo '    save_latex_pdf = True' >> "+filename_params)

    for filename in os.listdir(testdir):
        if filename.startswith('reference_Iapprox') and filename.endswith('.npy'):
            filename_Iapprox         = testdir + '/' + filename
        if filename.startswith('reference_Iexact') and filename.endswith('.npy'):
            filename_Iexact         = testdir + '/' + filename

    assert os.path.isfile(filename_params),  'params.py is missing.'
    assert os.path.isfile(filename_run),     'runscript.py is missing.'
    assert os.path.isfile(filename_Iexact),  'Iexact is missing.'
    assert os.path.isfile(filename_Iapprox), 'Iapprox is missing.'

    os.chdir(testdir)

    os.system('python3 runscript.py')

    for filename in os.listdir(testdir):
        if filename.startswith('Iapprox') and filename.endswith('.npy'):
            filename_Iapprox_printed = testdir + '/' + filename
        if filename.startswith('Iexact') and filename.endswith('.npy'):
            filename_Iexact_printed = testdir + '/' + filename

    assert os.path.isfile(filename_Iexact_printed),  "Iexact is not printed from the code"
    assert os.path.isfile(filename_Iapprox_printed), "Iapprox is not printed from the code"

    # Load all relevant files and restrict data to max 10th order
    Iexact_reference     = np.array(np.load(filename_Iexact))
    freqw = Iexact_reference[3]
    # All indices between 0 and 10th order
    freq_idx = np.where(np.logical_and(0 <= freqw, freqw <= 10))[0]

    Iexact_E_dir_reference = Iexact_reference[6][freq_idx]
    Iexact_ortho_reference = Iexact_reference[7][freq_idx]

    Iexact_printed       = np.array(np.load(filename_Iexact_printed))
    Iexact_E_dir_printed = Iexact_printed[6][freq_idx]
    Iexact_ortho_printed = Iexact_printed[7][freq_idx]

    Iapprox_reference       = np.array(np.load(filename_Iapprox))
    Iapprox_E_dir_reference = Iapprox_reference[6][freq_idx]
    Iapprox_ortho_reference = Iapprox_reference[7][freq_idx]

    Iapprox_printed       = np.array(np.load(filename_Iapprox_printed))
    Iapprox_E_dir_printed = Iapprox_printed[6][freq_idx]
    Iapprox_ortho_printed = Iapprox_printed[7][freq_idx]

    print("\n\nMaxima of the emission spectra: ",
          "\nexact  E_dir: ", np.amax(np.abs(Iexact_E_dir_reference))  ,
          "\nexact  ortho: ", np.amax(np.abs(Iexact_ortho_reference))  ,
          "\napprox E_dir: ", np.amax(np.abs(Iapprox_E_dir_reference)) ,
          "\napprox ortho: ", np.amax(np.abs(Iapprox_ortho_reference)) )

    Iexact_relerror      = (np.abs(Iexact_E_dir_printed    + Iexact_ortho_printed   ) + 1.0E-90) / \
                           (np.abs(Iexact_E_dir_reference  + Iexact_ortho_reference ) + 1.0E-90) - 1
    Iapprox_relerror     = (np.abs(Iapprox_E_dir_printed   + Iapprox_ortho_printed  ) + 1.0E-90) / \
                           (np.abs(Iapprox_E_dir_reference + Iapprox_ortho_reference) + 1.0E-90) - 1

    Iexact_max_relerror  = np.amax(np.abs(Iexact_relerror))
    Iapprox_max_relerror = np.amax(np.abs(Iapprox_relerror))

    print("\n\nTesting the exact emission spectrum I(omega):",
      "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", Iexact_max_relerror,
        "\nThe threshold is:                                                                 ", threshold_rel_error, "\n")

    assert Iexact_max_relerror < threshold_rel_error, "The exact emission spectrum is not matching."

    print("Testing the approx. emission spectrum I(omega):",
      "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", Iapprox_max_relerror,
        "\nThe threshold is:                                                                 ", threshold_rel_error, "\n")

    assert Iapprox_max_relerror < threshold_rel_error, "The approx. emission spectrum is not matching."

    shutil.rmtree(testdir + '/__pycache__')
    for E0_dirname   in glob.glob(testdir + '/E0*'):   shutil.rmtree(E0_dirname)
    for PATH_dirname in glob.glob(testdir + '/PATH*'): shutil.rmtree(PATH_dirname)

    os.remove(testdir + '/' + glob.glob("Iexact_*")[0])
    os.remove(testdir + '/' + glob.glob("Iapprox_*")[0])
    for params_output in glob.glob(testdir + '/params_*'): os.remove(params_output)

    if print_latex_pdf_really:
        assert os.path.isfile(filename_pdf),  "The latex PDF is not there."
        os.system("sed -i '$ d' "+filename_params)
#        filename_pdf_final  = testdir + '/CUED_summary_current_version.pdf'
#        shutil.move(filename_pdf, filename_pdf_final)
        shutil.rmtree(testdir + '/latex_pdf_files')

    print('Test passed successfully.'
          '\n\n=====================================================\n\n')

    os.chdir("..")


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

    count = 0

    for subdir, dirs, files in os.walk(dirpath+'/tests'):
        for dir in sorted(dirs):
            testdir = os.path.join(subdir, dir)
            count += 1
            check_test(testdir)

    assert count > 0, 'There are no test files with ending .reference in directory ./' + testdir

if __name__ == "__main__":
    main()
