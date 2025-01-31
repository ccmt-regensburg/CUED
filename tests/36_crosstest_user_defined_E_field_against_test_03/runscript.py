from params import params

from numba import njit
import numpy as np
import cued.hamiltonian
from cued.main import sbe_solver
from cued.utility import ConversionFactors as CoFa

def make_electric_field_in_path(E0, f, sigma, chirp, phase):
    """
    Creates a jitted version of the electric field for fast use inside a solver
    """
    @njit
    def electric_field(t):
        '''
        Returns the instantaneous driving pulse field
        '''
        # Non-pulse
        # return E0*np.sin(2.0*np.pi*w*t)
        # Chirped Gaussian pulse
        return E0*np.exp(-t**2/sigma**2) \
            * np.sin(2.0*np.pi*f*t*(1 + chirp*t) + phase)

    return electric_field


def dirac():
	A = 0.1974      # Fermi velocity

	dirac_system = cued.hamiltonian.BiTe(C0=0, C2=0, A=A, R=0, mz=0)

	return dirac_system

def run(system):

	E0                = 5.00*CoFa.MVpcm_to_au                     # Pulse amplitude (MV/cm)
	f                 = 25.0*CoFa.THz_to_au                     # Pulse frequency (THz)
	chirp             = 0.00                     # Pulse chirp ratio (chirp = c/w) (THz)
	sigma             = 50.0*CoFa.fs_to_au                     # Gaussian pulse width (femtoseconds)
	phase             = 0.0

	params.electric_field_function_in_path = make_electric_field_in_path(E0, f, sigma, chirp, phase)
	params.electric_field_function_ortho = make_electric_field_in_path(0, f, sigma, chirp, phase) # orthogonal field is zero

	sbe_solver(system, params)

	return 0

if __name__ == "__main__":
	run(dirac())
