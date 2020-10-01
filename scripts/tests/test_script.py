import os
import argparse
import numpy as np

####################################################################################################
#THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py#
####################################################################################################

def check_test(testdir, filename_reference, args):

    threshold_rel_error = 1.0E-10
    threshold_abs_error = 1.0E-24
    filename = './' + testdir + '/' + 'test.dat'
    filename_reference = './' + testdir + '/' + filename_reference

    assert os.path.isfile(filename_reference), 'Reference file is missing.'

    print('\n\n=====================================================\n\nStart with test:'
          '\n\n' + filename_reference +
          '\n\n=====================================================\n\n'
          'Output from the script:\n')

   # first line in filename_reference is the command to execute the code
    with open(filename_reference) as f:
        first_line = f.readline()
        os.system(first_line)

    assert os.path.isfile(filename), "Testfile is not printed from the code"

    print('\n=====================================================')
    print('\nRelative-error threshold: ' + str(threshold_rel_error) + ', absolute-error threshold: '
          + str(threshold_abs_error) +
          ' (One of both thresholds needs to be satisfied)')
    print('\nThe following numbers are tested:\n')
    print('{:<15} {:<25} {:<25} {:<25} {:<25}'.format('Quantity', 'Reference number', \
                                                      'Number from current git', 'Relative error',
                                                      'Absolute error'))

    # reset mode of the script
    if args.reset:
        os.system('sed -n -i \'1p\' ' + filename_reference)
        os.system('cat ' + filename + " >> " + filename_reference)

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

    testdir = 'scripts/tests'
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('-reset', default=False, action='store_true',
                        help=('Flag to reset all *.reference files in ./' + testdir +
                              '. Needed: Put all .reference files you want to reset/update'
                              'in ./' + testdir  + 'and insert the command to execute the'
                              ' main script in the first line of'
                              'the .reference file. The reset mode of this script'
                              'will insert the lines of the test file after the first'
                              'line (which contains the command to execute the main'
                              'script).'))
    args = parser.parse_args()

    is_dir = os.path.isdir('./' + testdir)
    _dirpath = os.getcwd()

    assert is_dir, 'The directory ./' + testdir + ' does not exist.'

    count = 0
    for filename_reference in os.listdir('./' + testdir):
        if filename_reference.endswith(".reference"):
            check_test(testdir, filename_reference, args)
        assert count > 0, 'There are no test files with ending .reference in directory ./' + testdir

if __name__ == "__main__":
    main()
