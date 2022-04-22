# Input parameters for SBE.py
import numpy as np


class params:
	# System parameters
	#########################################################################
	e_fermi             = 0.0                   # Fermi energy in eV
	temperature         = 0.00                  # Temperature in eV

	# Model Hamiltonian parameters
	# Brillouin zone parameters
	##########################################################################
	# Type of Brillouin zone
	BZ_type             = 'rectangle'

	# rectangle BZ parameters
	# for Fig. 1b in Paper one has to set Nk1 = 1200 and Nk2 = number of paths
	Nk1                 = 50                     # Number of kpoints in each of the paths
	Nk2                 = 4                      # Number of paths
	length_BZ_E_dir     = 1.15                   # length of BZ in E-field direction
	length_BZ_ortho     = 0.5                    # length of BZ orthogonal to E-field direction
	angle_inc_E_field   = 0                      # incoming angle of the E-field in degree

	# Driving field parameters
	##########################################################################
	f                      = 50.0                # Pulse frequency (THz)
	factor_freq_resolution = 2

	# Time scales (all units in femtoseconds)
	##########################################################################
	T1                  = 1000                   # Phenomenological diagonal damping time
	T2                  = 10                     # Phenomenological polarization damping time
	t0                  = -1000                  # Start time *pulse centered @ t=0, use t0 << 0
	dt                  = 0.1                    # Time step

	# Flags for testing and features
	##########################################################################
	gauge                   = 'length'           # Gauge of the system
	solver                  = '2band'
	fourier_window_function = 'hann'
	user_out                = False
	save_latex_pdf 			= False
