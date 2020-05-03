from numba import njit
import numpy as np


def make_electric_field(E0, w, alpha, chirp, phase):
    @njit
    def electric_field(t):
        '''
        Returns the instantaneous driving pulse field
        '''
        # Non-pulse
        # return E0*np.sin(2.0*np.pi*w*t)
        # Chirped Gaussian pulse
        return E0*np.exp(-t**2.0/(2.0*alpha)**2) \
            * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

    return electric_field


def make_zeeman_field(B0, mu, w, alpha, chirp, phase, E_dir, incident_angle):
    @njit
    def zeeman_field(t):
        time_dep = np.exp(-t**2.0/(2.0*alpha)**2) \
            * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

        # x, y, z components
        m_zee = np.empty(3)
        m_zee[0] = mu[0]*B0*E_dir[1] * np.cos(incident_angle) * time_dep
        m_zee[1] = mu[1]*B0*E_dir[0] * np.cos(incident_angle) * time_dep
        m_zee[2] = mu[2]*B0*np.sin(incident_angle) * time_dep

        return m_zee

    return zeeman_field

# @njit
# def curl_electric_field(mdx, mdy, mdz, w, t, chirp, alpha, phase, E_dir,
#                         incident_angle):
#     time_dep = np.exp(-t**2.0/(2.0*alpha)**2) \
#         * np.cos(2.0*np.pi*w*t*(1 + chirp*t) + phase)
#     mx = mdx * E_dir[1] * np.cos(incident_angle) * time_dep
#     my = -mdy * E_dir[0] * np.cos(incident_angle) * time_dep
#     mz = mdz * np.sin(incident_angle) * time_dep
#     return mx, my, mz
