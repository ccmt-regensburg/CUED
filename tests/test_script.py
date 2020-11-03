import os
import argparse
import numpy as np
import shutil
import sbe.plotting as splt

####################################################################################################
#THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py#
####################################################################################################

def check_test(testdir):

    print('\n\n=====================================================\n\nStart with test:'
          '\n\n' + testdir +
          '\n\n=====================================================\n\n')

    threshold_rel_error = 1.0E-15

    filename_params  = testdir + '/params.py'
    filename_run     = testdir + '/runscript.py' 

    for filename in os.listdir(testdir):
        if filename.startswith('reference_Iapprox'):
            filename_Iapprox         = testdir + '/' + filename 
            filename_Iapprox_printed = testdir + '/' + filename[10:]

        if filename.startswith('reference_Iexact'):
            filename_Iexact         = testdir + '/' + filename 
            filename_Iexact_printed = testdir + '/' + filename[10:]

    assert os.path.isfile(filename_params),  'params.py is missing.'
    assert os.path.isfile(filename_run),     'runscript.py is missing.'
    assert os.path.isfile(filename_Iexact),  'Iexact is missing.'
    assert os.path.isfile(filename_Iapprox), 'Iapprox is missing.'

    os.chdir(testdir)

    os.system('python3 runscript.py')

    print('\n\nPythonpath:           ', os.environ['PYTHONPATH'].split(os.pathsep))
    print('\n\nFiles in testdir:     ', os.listdir(testdir))
    print('\n\nFiles in testdir/../: ', os.listdir(testdir+'/..'))

    assert os.path.isfile(filename_Iexact_printed),  "Iexact is not printed from the code"
    assert os.path.isfile(filename_Iapprox_printed), "Iapprox is not printed from the code"

    print('\n=====================================================\n')

    Iexact_reference = np.array(np.load(filename_Iexact))
    Iexact_printed   = np.array(np.load(filename_Iexact_printed))

    Iexact_relerror  = (Iexact_printed[6].real+1.0E-90) / (Iexact_reference[6].real+1.0E-90) - 1

    max_relerror = np.amax(np.absolute(Iexact_relerror))

    print("The maximum relative deviation is:", max_relerror, 
      "\n\nThe threshold is:                 ", threshold_rel_error)


    assert np.amax(np.absolute(Iexact_relerror)) < threshold_rel_error, "The emission spectrum is not matching."

    shutil.rmtree(testdir+'/__pycache__')

    print('\nTest passed successfully.'
          '\n\n=====================================================\n\n')

    os.chdir("..")

    # reset mode of the script
    #if args.reset:
    #    os.system('sed -n -i \'1p\' ' + filename_reference)
    #    os.system('cat ' + filename + " >> " + filename_reference)

    # # normal test mode if the script
    # else:
    #   with open(filename) as f:
    #       count = 0
    #       for line in f:
    #           count += 1
    #           fields = line.split()
    #           value_test = float(fields[1])

    #           with open(filename_reference) as f_reference:

    #               count_reference = 0
    #               for line_reference in f_reference:
    #                   count_reference += 1
    #                   fields_reference = line_reference.split()

    #                   # we have the -1 because there is the header with executing command
    #                   # in the reference file
    #                   if count == count_reference-1:
    #                       value_reference = float(fields_reference[1])

    #                       abs_error = np.abs(value_test - value_reference)
    #                       rel_error = abs_error/np.abs(value_reference)

    #                       check_abs = abs_error < threshold_abs_error
    #                       check_rel = rel_error < threshold_rel_error

    #                       print('{:<15} {:>25} {:>25} {:>25} {:>25}'.format(fields_reference[0], \
    #                             value_reference, value_test, rel_error, abs_error))

    #                       assert check_abs or check_rel, \
    #                              "\n\nAbsolute and relative error of variable number "+str(count)+\
    #                              " compared to reference too big:"\
    #                              "\n\nRelative error: "+str(rel_error)+" and treshold: "+str(threshold_rel_error)+\
    #                              "\n\nAbsolute error: "+str(abs_error)+" and treshold: "+str(threshold_abs_error)

    #           f_reference.close()

    #   print("\n\nTest passed successfully.\n\n")

    #   f.close()

def main():

#    parser = argparse.ArgumentParser(description='Test script')
#    parser.add_argument('-reset', default=False, action='store_true',
#                        help=('Flag to reset all *.reference files in ./' + testdir +
#                              '. Needed: Put all .reference files you want to reset/update'
#                              'in ./' + testdir  + 'and insert the command to execute the'
#                              ' main script in the first line of'
#                              'the .reference file. The reset mode of this script'
#                              'will insert the lines of the test file after the first'
#                              'line (which contains the command to execute the main'
#                              'script).'))
#    args = parser.parse_args()


    dirpath = os.getcwd()

    print('\n\n=====================================================\n\n SBE CODE TESTER'
          '\n\n Executed in the directory:\n\n '+dirpath+
          '\n\n=====================================================\n\n')

    count = 0

    for subdir, dirs, files in os.walk(dirpath+'/tests'):
        for dir in dirs:
            testdir = os.path.join(subdir, dir)
            count += 1
            check_test(testdir)

    assert count > 0, 'There are no test files with ending .reference in directory ./' + testdir

if __name__ == "__main__":
    main()
