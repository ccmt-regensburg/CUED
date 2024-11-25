import numpy as np
from cued.utility import conditional_njit

def make_electric_field_in_path(P):
    """
    Creates a jitted version of the electric field for fast use inside a solver
    """
    E0 = P.E0
    f = P.f
    sigma =  P.sigma
    chirp = P.chirp
    phase = P.phase

    @conditional_njit(P.type_real_np)
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

def make_electric_field_ortho(P):
    """
    Creates a jitted version of the electric field for fast use inside a solver
    """
    E0_ort = P.E0_ort
    f_ort = P.f_ort
    sigma_ort =  P.sigma_ort
    chirp_ort = P.chirp_ort
    phase_ort = P.phase_ort

    @conditional_njit(P.type_real_np)
    def electric_field(t):
        '''
        Returns the instantaneous driving pulse field
        '''
        # Non-pulse
        # return E0*np.sin(2.0*np.pi*w*t)
        # Chirped Gaussian pulse
        return E0_ort*np.exp(-t**2/sigma_ort**2) \
            * np.sin(2.0*np.pi*f_ort*t*(1 + chirp_ort*t) + phase_ort)

    return electric_field

