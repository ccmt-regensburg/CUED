# Input parameters for SBE.py
import numpy as np

# Variable for test_script.py
MPI_JOBS=8


class params:
	# System parameters
	#########################################################################
	a                   = 8.308       # Lattice spacing in atomic units (4.395 A)
	e_fermi             = 0.00        # Fermi energy in eV
	temperature         = 0.03        # Temperature in eV

	# Model Hamiltonian parameters
	# Brillouin zone parameters
	##########################################################################
	# Type of Brillouin zone
	# 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
	BZ_type = 'hexagon'

	# full BZ parametes
	Nk1                 = 420         # Number of kpoints in each of the paths
	Nk2                 = 42          # Number of paths

	# Driving field parameters
	##########################################################################
	align               = 'M'         # E-field direction (gamma-'K' or gamma-'M')
	E0                  = 3.0         # Pulse amplitude (MV/cm)
	f                   = 25.0        # Pulse frequency (THz)
	chirp               = -1.25       # Pulse chirp ratio (chirp = c/w) (THz)
	sigma               = 90          # Gaussian pulse width (femtoseconds)
	phase               = np.linspace(0, 2*np.pi, 16)

	# Time scales (all units in femtoseconds)
	##########################################################################
	T1    = 1000     # Phenomenological diagonal damping time
	T2    = 10       # Phenomenological polarization damping time
	t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
	dt    = 5e-2     # Time step

	# Flags for testing and features
	##########################################################################
	gauge                   = 'length'   # Gauge of the system
	solver                  = '2band'
	fourier_window_function = 'gaussian'
	user_out                = False
	plot_format             = 'png'
