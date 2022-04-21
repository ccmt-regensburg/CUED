from numpy import pi
import scipy.constants as sco
from scipy.constants import physical_constants as psco

class ConversionFactors:
	'''
	Collection of conversion factors from SI to atomic units
	'''
	##################################################
	# SI <-> a.u. conversions
	##################################################
	# Better accuracy -> Problems with test files
	# (1fs = 41.3414733358211 a.u.)
	# fs_to_au = sco.femto/psco['atomic unit of time'][0]
	# au_to_fs = psco['atomic unit of time'][0]/sco.femto

	# (1fs = 41.341473335 a.u.)
	fs_to_au = 41.34137335
	au_to_fs = 1/fs_to_au

	# (1MV/cm = 1.944690381*10^-4 a.u.)
	MVpcm_to_au = 0.0001944690381
	au_to_MVpcm = 1/MVpcm_to_au

	# (1THz   = 2.4188843266*10^-5 a.u.)
	THz_to_au = 0.000024188843266
	au_to_THz = 1/THz_to_au

	# (1A     = 150.97488474)
	Amp_to_au = 150.97488474
	au_to_Amp = 1/Amp_to_au

	# (1eV    = 0.036749322176 a.u.)
	eV_to_au = 0.03674932176
	au_to_eV = 1/eV_to_au

	# (1Angst.= 1.8897261254535 a.u.)
	as_to_au = 1.8897261254535
	au_to_as = 1/as_to_au

	##################################################
	# Magnetic field constants
	##################################################
	# (1T = 4.25531e-6 a.u.)
	T_to_au = 4.25531e-6
	au_to_T = 1/T_to_au

	# (1mu_b = 0.5 a.u.)
	muB_to_au = 0.5
	au_to_muB = 1/muB_to_au

	##################################################
	# SI conversions
	##################################################
	## Frequency conversions
	# (1 eV = 241.7991 THz)
	eV_to_THz = 241.7991
	THz_to_eV = 1/eV_to_THz

	# (1 eV = 0.2417991 PHz) Petahertz is relevant for T2 times in fs
	eV_to_PHz = 0.2417991
	PHz_to_eV = 1/eV_to_PHz

	## Angular Frequency conversions
	eV_to_angular_THz = 2*pi*eV_to_THz
	angular_THz_to_eV = 1/eV_to_angular_THz

	eV_to_angular_PHz = 2*pi*eV_to_PHz
	angular_PHz_to_eV = 1/eV_to_angular_PHz

