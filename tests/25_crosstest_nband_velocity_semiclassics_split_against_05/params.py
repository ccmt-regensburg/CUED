# Input parameters for SBE.py
import numpy as np


class params:
	# System parameters
	#########################################################################
	e_fermi             = 0.0                    # Fermi energy in eV
	temperature         = 0.0                    # Temperature in eV

	# Model Hamiltonian parameters
	# Brillouin zone parameters
	##########################################################################
	# Type of Brillouin zone
	BZ_type             = 'rectangle'            # rectangle or hexagon
	Nk1                 = 10                      # Number of kpoints in each of the paths
	Nk2                 = 2                      # Number of paths
	length_BZ_E_dir     = 5.0                    # length of BZ in E-field direction
	length_BZ_ortho     = 0.1                    # length of BZ orthogonal to E-field direction
	angle_inc_E_field   = 0                      # incoming angle of the E-field in degree

	# Driving field parameters
	##########################################################################
	E0                  = 5.0                    # Pulse amplitude (MV/cm)
	f                   = 25.0                   # Pulse frequency (THz)
	chirp               = 0.00                   # Pulse chirp ratio (chirp = c/w) (THz)
	sigma               = 50.0                   # Gaussian pulse width (femtoseconds)
	phase               = 0.0
	angle_inc_E_field   = 0                      # incoming angle of the E-field in degree

	# Time scales (all units in femtoseconds)
	##########################################################################
	T1                  = 1000                   # Phenomenological diagonal damping time
	T2                  = 1                      # Phenomenological polarization damping time
	t0                  = -1000                  # Start time *pulse centered @ t=0, use t0 << 0
	dt                  = 0.05                   # Time step

	# Flags for testing and features
	##########################################################################
	gauge                   = 'velocity'         # Gauge of the system
	solver                  = 'nband'
	fourier_window_function = 'gaussian'
	split_current			= True
	dm_dynamics_method      = 'semiclassics'
	save_anom				= True
	user_out                = False
	save_latex_pdf = True
