import numpy as np
from numba import njit


def make_electric_field(E0, w, alpha, chirp, phase):
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
        return E0*np.exp(-t**2/(2*alpha)**2) \
            * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

    return electric_field


def electric_field(t, E0, w, alpha, chirp, phase):
    """
    Normal numpy version of a electric field
    """
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    return E0*np.exp(-t**2/(2*alpha)**2) \
        * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)
