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
	BZ_type             = 'rectangle'
	Nk1                 = 25                      # Number of kpoints in each of the paths
	Nk2                 = 2                      # Number of paths
	length_BZ_E_dir     = 0.5                    # length of BZ in E-field direction
	length_BZ_ortho     = 0.1                    # length of BZ orthogonal to E-field direction
	angle_inc_E_field   = 0                      # incoming angle of the E-field in degree

	# Driving field parameters
	##########################################################################
	E0                  = 5.00                   # Pulse amplitude (MV/cm)
	f                   = 25.0                   # Pulse frequency (THz)
	chirp               = 0.00                   # Pulse chirp ratio (chirp = c/w) (THz)
	sigma               = 35.0                   # Gaussian pulse width (femtoseconds)
	phase               = 0.0

	# Time scales (all units in femtoseconds)
	##########################################################################
	T1                  = 1000                   # Phenomenological diagonal damping time
	T2                  = 10                      # Phenomenological polarization damping time
	t0                  = -400                  # Start time *pulse centered @ t=0, use t0 << 0
	dt                  = 0.5                   # Time step

	# Flags for testing and features
	##########################################################################
	gauge                   = 'velocity'          # Gauge of the system
	solver                  = '2band'
	fourier_window_function = 'gaussian'
	user_out                = True               # True to get user plotting and progress output
	gabor_transformation	= True
	gabor_gaussian_center	= np.linspace(-100,100,num=4)
	gabor_window_width      = [10,35]
	save_latex_pdf          = False
