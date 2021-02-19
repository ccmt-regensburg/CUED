import numpy as np
from sbe.utility import conditional_njit

def make_electric_field(P):
    """
    Creates a jitted version of the electric field for fast use inside a solver
    """
    E0 = P.E0
    w = P.w
    alpha = P.alpha
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
        return E0*np.exp(-t**2/(2*alpha)**2) \
            * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

    return electric_field
