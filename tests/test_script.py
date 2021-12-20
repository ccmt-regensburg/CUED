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
threshold_rel_error	= 0.1

time_suffix = "time_data.dat"
freq_suffix = "frequency_data.dat"
params_suffix = "params.txt"
pdf_suffix = "latex_pdf_files"

def check_test(testdir, refdir):

	print('\n\n=====================================================\n\nStart with test:'
	      '\n\n' + testdir + '\n\n against reference in \n\n' + refdir + '\n\n')

	filename_params		= testdir + '/params.py'
	filename_run		= testdir + '/runscript.py'

	params, current_mpi_jobs = import_params(filename_params)

	# Get name of files (needed for MPI-tests)
	time_filenames = glob.glob1(refdir, "*" + time_suffix)
	refe_prefixes = [time_filename.replace(time_suffix, "") for time_filename in time_filenames]
	test_prefixes = [prefix.replace('reference_', '') for prefix in refe_prefixes]
	pdf_foldernames = [prefix + pdf_suffix for prefix in test_prefixes]

	if hasattr(params,"gabor_transformation") and params.gabor_transformation == True:
		gabor_filenames = glob.glob1(refdir, "reference_gabor"+"*" + freq_suffix)
		gabor_refe_prefixes = [freq_filename.replace(freq_suffix, "") for freq_filename in gabor_filenames]
		gabor_test_prefixes = [prefix.replace('reference_', '') for prefix in gabor_refe_prefixes]

	print_latex_pdf_really = check_params_for_print_latex_pdf(print_latex_pdf, params)

	if print_latex_pdf_really:
		os.system("echo '	save_latex_pdf = True' >> "+filename_params)

	assert os.path.isfile(filename_params),	 'params.py is missing.'
	assert os.path.isfile(filename_run),	 'runscript.py is missing.'

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
	os.system('mpirun -n ' + str(current_mpi_jobs) + ' python3 -W ignore ' + testdir + '/runscript.py')
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

		build_comparable_data(i,freq_data,freq_data_ref)

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

			build_comparable_data(i,gabor_freq_data,gabor_freq_data_ref)


	shutil.rmtree(testdir + '/__pycache__')
	for E0_dirname	 in glob.glob(testdir + '/E0*'):   shutil.rmtree(E0_dirname)
	for PATH_dirname in glob.glob(testdir + '/PATH*'): shutil.rmtree(PATH_dirname)

	print('Test passed successfully.'
	      '\n\n=====================================================\n\n')


def read_data(dir, prefix):

	# Reading in reference data
	time_data, freq_data, _dens_data = read_dataset(dir, prefix=prefix, mute=True)

	return time_data, freq_data


def build_comparable_data(freq_data_index,freq_data,freq_data_ref):
	# Load all relevant files and restrict data to max 10th order
	freq = freq_data['f/f0']

	# All indices between 0 and 10th order
	freq_idx = np.where(np.logical_and(0 <= freq, freq <= 10))[0]

	# Emission
	I_E_dir_ref = freq_data_ref[freq_data_index]['I_E_dir'][freq_idx]
	I_ortho_ref = freq_data_ref[freq_data_index]['I_ortho'][freq_idx]
	I_E_dir = freq_data['I_E_dir'][freq_idx]
	I_ortho = freq_data['I_ortho'][freq_idx]
	print("\n\nMaxima of the emission spectra: ",
		"\nfull	 E_dir: ", np.amax(np.abs(I_E_dir_ref)),
		"\nfull	 ortho: ", np.amax(np.abs(I_ortho_ref)))
	check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, 'full')


def check_emission(I_E_dir, I_ortho, I_E_dir_ref, I_ortho_ref, name):
	relerror = (np.abs(I_E_dir + I_ortho) + 1.0E-90) / \
	           (np.abs(I_E_dir_ref + I_ortho_ref) + 1.0E-90) - 1

	max_relerror = np.amax(np.abs(relerror))

	print("\n\nTesting the \"" + name + "\" emission spectrum I(omega):",
	      "\n\nThe maximum relative deviation between the computed and the reference spectrum is:", max_relerror,
	      "\nThe threshold is:																 ", threshold_rel_error, "\n")

	assert max_relerror < threshold_rel_error, "The \"" + name + "\" emission spectrum is not matching."


def import_params(filename_params):
	"""
	Imports the file dependent parameter file. If MPI_JOBS is set
	it changes the number of started jobs for the file.
	"""

	spec = importlib.util.spec_from_file_location("params", filename_params)
	params = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(params)
	if hasattr(params, 'MPI_JOBS'):
		current_mpi_jobs = params.MPI_JOBS
	else:
		current_mpi_jobs = default_mpi_jobs

	return params.params(), current_mpi_jobs

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

def main(testpath):

	tests_path = HEADPATH + '/' + testpath
	print('\n\n=====================================================\n\n CUED CODE TESTER'
	      '\n\n Executing tests in:\n\n '+ tests_path +
	      '\n\n=====================================================\n\n')

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
	parser.add_argument("-n", "--mpin", type=int, help="Number of mpi jobs")
	args = parser.parse_args()

	return args.path, args.latex, args.mpin

if __name__ == "__main__":
	path, print_latex_pdf, default_mpi_jobs = parser()
	main(path)
