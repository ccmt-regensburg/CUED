import os
import glob
import argparse
import numpy as np
import shutil
import importlib.util
import importlib.machinery
import sys

from cued.plotting import read_dataset
from cued.utility import ParamsParser

######################################################################################################
# THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py #
######################################################################################################

#################################
# PARAMETERS OF THE TEST SCRIPT #
#################################
print_latex_pdf		  = False
threshold_rel_error	  = 0.1


def check_test(testdir, refdir):

	print('\n\n=====================================================\n\nStart with test:'
		  '\n\n' + testdir + '\n\n against reference in \n\n' + refdir + '\n\n')

	filename_params		= testdir + '/params.py'
	filename_run		= testdir + '/runscript.py'
	filename_pdf		= testdir + '/latex_pdf_files/CUED_summary.pdf'

	params = import_params(filename_params)

	# Get name of files (needed for MPI-tests)
	filenames = glob.glob1(refdir, "*frequency_data.dat")
	number_of_files = len(filenames)

	print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

	if print_latex_pdf_really:
		os.system("echo '    save_latex_pdf = True' >> "+filename_params)

	assert os.path.isfile(filename_params),	 'params.py is missing.'
	assert os.path.isfile(filename_run),	 'runscript.py is missing.'

	time_data_ref = []
	freq_data_ref = []

	for i in range(number_of_files):

		prefix = filenames[i][:-18]
		time_data_tmp, freq_data_tmp, _dens_data = read_dataset(refdir, prefix=prefix)

		assert time_data_tmp is not None, 'Reference time_data is missing.'
		assert freq_data_tmp is not None, 'Reference frequency_data is missing.'

		time_data_ref.append(time_data_tmp)
		freq_data_ref.append(freq_data_tmp)

	os.chdir(testdir)

	os.system('mpirun -np 2 python3 runscript.py')

	# Reading in generated data
	for i in range(number_of_files):
		prefix = filenames[i][10:-18]
		time_data, freq_data, _dens_data = read_dataset(os.getcwd(), prefix=prefix)

		assert time_data is not None, '"time_data.dat" was not generated from the code'
		assert freq_data is not None, '"frequency_data.dat" was not generated from the code'

		# Load all relevant files and restrict data to max 10th order
		freq = freq_data['f/f0']

		# All indices between 0 and 10th order
		freq_idx = np.where(np.logical_and(0 <= freq, freq <= 10))[0]

		# Emission
		I_E_dir_ref = freq_data_ref[i]['I_E_dir'][freq_idx]
		I_ortho_ref = freq_data_ref[i]['I_ortho'][freq_idx]
		I_E_dir = freq_data['I_E_dir'][freq_idx]
		I_ortho = freq_data['I_ortho'][freq_idx]
		print("\n\nMaxima of the emission spectra: ",
			"\nfull	 E_dir: ", np.amax(np.abs(I_E_dir_ref)),
			"\nfull	 ortho: ", np.amax(np.abs(I_ortho_ref)))
		check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, 'full')

		if hasattr(params, 'split_current'):
			if params.split_current:
				# Intra + dtP emission
				I_intra_plus_dtP_E_dir_ref = freq_data_ref[i]['I_intra_plus_dtP_E_dir'][freq_idx]
				I_intra_plus_dtP_ortho_ref = freq_data_ref[i]['I_intra_plus_dtP_E_dir'][freq_idx]
				I_intra_plus_dtP_E_dir = freq_data['I_intra_plus_dtP_E_dir'][freq_idx]
				I_intra_plus_dtP_ortho = freq_data['I_intra_plus_dtP_E_dir'][freq_idx]
				print("\nintra plus dtP E_dir: ", np.amax(np.abs(I_intra_plus_dtP_E_dir_ref)),
					"\nintra plus dtP ortho: ", np.amax(np.abs(I_intra_plus_dtP_ortho_ref)))
				check_emission(I_intra_plus_dtP_E_dir, I_intra_plus_dtP_ortho,
							I_intra_plus_dtP_E_dir_ref, I_intra_plus_dtP_ortho_ref,
							'dtP')

		if hasattr(params, 'save_anom'):
			if params.save_anom:
				# Intra + anom emission
				I_anom_ortho_ref = freq_data_ref[i]['I_anom_ortho'][freq_idx]
				I_anom_ortho = freq_data['I_anom_ortho'][freq_idx]

				print("\nintra plus dtP E_dir: ", np.amax(np.abs(I_anom_ortho_ref)))
				check_emission(I_anom_ortho, I_anom_ortho,
							I_anom_ortho_ref, I_anom_ortho_ref,
							'anom')

		os.remove(testdir + '/' + filenames[i][10:-18] + 'time_data.dat')
		os.remove(testdir + '/' + filenames[i][10:-18] + 'frequency_data.dat')
		os.remove(testdir + '/' + filenames[i][10:-18] + 'params.txt')

	shutil.rmtree(testdir + '/__pycache__')
	for E0_dirname	 in glob.glob(testdir + '/E0*'):   shutil.rmtree(E0_dirname)
	for PATH_dirname in glob.glob(testdir + '/PATH*'): shutil.rmtree(PATH_dirname)


	if print_latex_pdf_really:
		assert os.path.isfile(filename_pdf),  "The latex PDF is not there."
		os.system("sed -i '$ d' "+filename_params)
		filename_pdf_final	= testdir + '/CUED_summary_current_version.pdf'
		shutil.move(filename_pdf, filename_pdf_final)
		shutil.rmtree(testdir + '/latex_pdf_files')

	print('Test passed successfully.'
		  '\n\n=====================================================\n\n')

	os.chdir("..")
	sys.path.append("..")

def read_data(dir, prefix):

	# Reading in reference data
	time_data, freq_data, _dens_data = read_dataset(dir, prefix=prefix)

	return time_data, freq_data


def check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, name):
	relerror = (np.abs(I_E_dir + I_ortho) + 1.0E-90) / \
		(np.abs(I_E_dir_ref + I_ortho_ref) + 1.0E-90) - 1

	max_relerror = np.amax(np.abs(relerror))

	print("\n\nTesting the \"" + name + "\" emission spectrum I(omega):",
	  "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", max_relerror,
		"\nThe threshold is:																 ", threshold_rel_error, "\n")

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

def main(testpath):
	dirpath = os.getcwd()

	print('\n\n=====================================================\n\n SBE CODE TESTER'
		  '\n\n Executed in the directory:\n\n '+dirpath+
		  '\n\n=====================================================\n\n')

	tests_path = dirpath + '/' + testpath
	count = 0

	for cdir in sorted(os.listdir(tests_path)):
		testdir = tests_path + '/' + cdir
		if os.path.isdir(testdir) and not cdir.startswith('norun') and not cdir.startswith('crosstest', 3):
			count += 1
			check_test(testdir, testdir)
		if os.path.isdir(testdir) and cdir.startswith('crosstest', 3):
			for dummydir in sorted(os.listdir(tests_path)):
				if dummydir.startswith(cdir[-2:]):
					refdir = tests_path + '/' + dummydir
			check_test(testdir, refdir)

	assert count > 0, 'There are tests in directory ' + tests_path

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--path", type=str, help="Relative testpath with respect to top level CUED dir.")
	parser.add_argument("-l", "--latex", type=bool, help="Latex - PDF compilation.")
	args = parser.parse_args()

	return args.path, args.latex

if __name__ == "__main__":
	path, print_latex_pdf = parser()	
	main(path)
