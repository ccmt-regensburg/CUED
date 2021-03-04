import numpy as np
from cued.utility import conditional_njit

def make_electric_field(P):
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
        return E0*np.exp(-t**2/(2*sigma**2)) \
            * np.sin(2.0*np.pi*f*t*(1 + chirp*t) + phase)

    return electric_field
