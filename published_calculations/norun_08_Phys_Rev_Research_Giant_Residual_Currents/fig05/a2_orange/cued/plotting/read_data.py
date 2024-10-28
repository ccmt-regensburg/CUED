import numpy as np
import os

def read_datasets(subpaths, mute=False):
	"""
	Specify a path and read all subfolders
	read_dataset defines the individual order
	"""
	time_data_container = []
	freq_data_container = []
	dens_data_container = []

	for subpath in subpaths:
		if not mute:
			print("Evaluating " + subpath + " data", end='\n\n')
		time_data, freq_data, dens_data = read_dataset(subpath)
		time_data_container.append(time_data)
		freq_data_container.append(freq_data)
		dens_data_container.append(dens_data)

	return time_data_container, freq_data_container, dens_data_container

def read_dataset(path, prefix='', suffix='', mute=False):
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
			if not mute:
				print("Reading time data:", filepath)
			time_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

		# Frequency data
		if (filename.startswith(freq_string) and '.dat' in filename):
			if not mute:
				print("Reading frequency data:", filepath)
			freq_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

		# Density data
		if (filename.startswith(dens_string) and '.dat' in filename):
			if not mute:
				print("Reading density:", filepath)
			dens_data = np.genfromtxt(filepath, names=True, encoding='utf8', deletechars='')

	if time_data is None and freq_data is None and dens_data is None:
		raise RuntimeError("No Data found.")

	return time_data, freq_data, dens_data
