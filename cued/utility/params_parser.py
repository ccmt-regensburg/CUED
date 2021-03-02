from math import modf
import numpy as np
import sys

from cued.utility import ConversionFactors as co

def parse_params(user_params):
    class Params():
        pass

    P = Params()
    UP = user_params
    P.user_out = True                       # Command line progress output
    if hasattr(UP, 'user_out'):
        P.user_out = UP.user_out

    P.save_full = False                     # Save full density matrix
    if hasattr(UP, 'save_full'):
        P.save_full = UP.save_full

    P.split_current = False             # Save j^intra, j^anom, dP^inter/dt
    if hasattr(UP, 'split_current'):
        P.split_current = UP.split_current

    P.do_semicl = False                     # Semiclassical calc. (dipole = 0)
    if hasattr(UP, 'do_semicl'):
        P.do_semicl = UP.do_semicl

    P.gauge = 'length'                      # Gauge of the SBE Dynamics
    if hasattr(UP, 'gauge'):
        P.gauge = UP.gauge

    P.hamiltonian_evaluation = 'num'                        # numerical or analytical calculation of eigenstates and dipoles
    if hasattr(UP, 'hamiltonian_evaluation'):
        P.hamiltonian_evaluation = UP.hamiltonian_evaluation

    P.solver = 'nband'                      # 2 or n band solver
    if hasattr(UP, 'solver'):
        P.solver = UP.solver

    P.epsilon = 2e-5                        # step size for num. derivatives
    if hasattr(UP, 'epsilon'):
        P.epsilon = UP.epsilon

    P.gidx = 1
    if hasattr(UP, 'gidx'):
        P.gidx = UP.gidx

    P.save_anom = False
    if hasattr(UP, 'save_anom'):
        P.save_anom = UP.save_anom

    P.solver_method = 'bdf'                 # 'adams' non-stiff, 'bdf' stiff, 'rk4' Runge-Kutta 4th order
    if hasattr(UP, 'solver_method'):
        P.solver_method = UP.solver_method

    P.precision = 'double'                  # quadruple for reducing numerical noise
    if hasattr(UP, 'precision'):
        P.precision = UP.precision

    if P.precision == 'double':
        P.type_real_np    = np.float64
        P.type_complex_np = np.complex128
    elif P.precision == 'quadruple':
        P.type_real_np    = np.float128
        P.type_complex_np = np.complex256
        if P.solver_method != 'rk4':
            sys.exit("Error: Quadruple precision only works with Runge-Kutta 4 ODE solver.")
    else:
        sys.exit("Only default or quadruple precision available.")

    P.symmetric_insulator = False           # special flag for accurate insulator calc.
    if hasattr(UP, 'symmetric_insulator'):
        P.symmetric_insulator = UP.symmetric_insulator

    P.dk_order = 8
    if hasattr(UP, 'dk_order'):             # Accuracy order of density-matrix k-deriv.
        P.dk_order = UP.dk_order            # with length gauge (avail: 2,4,6,8)
        if P.dk_order not in [2, 4, 6, 8]:
            sys.exit("dk_order needs to be either 2, 4, 6, or 8.")

    # Parameters for initial occupation
    P.e_fermi = UP.e_fermi*co.eV_to_au           # Fermi energy
    P.e_fermi_eV = UP.e_fermi
    P.temperature = UP.temperature*co.eV_to_au   # Temperature
    P.temperature_eV =  UP.temperature

    # Driving field parameters
    P.E0 = UP.E0*co.MVpcm_to_au                  # Driving pulse field amplitude
    P.E0_MVpcm = UP.E0
    P.f = UP.f*co.THz_to_au                      # Driving pulse frequency
    P.f_THz = UP.f
    P.chirp = UP.chirp*co.THz_to_au              # Pulse chirp frequency
    P.chirp_THz = UP.chirp
    P.alpha = UP.alpha*co.fs_to_au               # Gaussian pulse width
    P.alpha_fs = UP.alpha
    P.phase = UP.phase                           # Carrier-envelope phase

    # Time scales
    P.T1 = UP.T1*co.fs_to_au                     # Occupation damping time
    P.gamma1 = 1/P.T1
    P.T1_fs = UP.T1
    P.gamma1_dfs = 1/P.T1_fs

    P.T2 = UP.T2*co.fs_to_au                     # Polarization damping time
    P.gamma2 = 1/P.T2
    P.T2_fs = UP.T2
    P.gamma2_dfs = 1/P.T2_fs

    P.t0 = UP.t0*co.fs_to_au
    P.t0_fs = UP.t0

    P.tf = -P.t0
    P.tf_fs = -P.t0_fs

    P.dt = P.type_real_np(UP.dt*co.fs_to_au)
    P.dt_fs = UP.dt

    Nf = int((abs(2*P.t0_fs))/P.dt_fs)
    if modf((2*P.t0_fs/P.dt_fs))[0] > 1e-12:
        print("WARNING: The time window divided by dt is not an integer.")
    # Define a proper time window if Nt exists
    # +1 assures the inclusion of tf in the calculation
    P.Nt = Nf + 1

    P.factor_freq_resolution = 1
    if hasattr(UP, 'factor_freq_resolution'):
        P.factor_freq_resolution = UP.factor_freq_resolution

    P.fourier_window_function = 'gaussian'     # gaussian or hann
    if hasattr(UP, 'fourier_window_function'):
        P.fourier_window_function = UP.fourier_window_function

    P.gaussian_window_width = P.alpha
    if hasattr(UP, 'gaussian_window_width'):
        P.gaussian_window_width = UP.gaussian_window_width

    # Brillouin zone type
    P.BZ_type = UP.BZ_type                      # Type of Brillouin zone
    P.Nk1 = UP.Nk1                              # kpoints in b1 direction
    P.Nk2 = UP.Nk2                              # kpoints in b2 direction
    P.Nk = P.Nk1 * P.Nk2

    # special parameters for individual Brillouin zone types
    if P.BZ_type == 'hexagon':
        P.align = UP.align                      # E-field alignment
        P.angle_inc_E_field = None
        P.a = UP.a                                  # Lattice spacing
        P.a_angs = P.a*co.au_to_as
    elif P.BZ_type == 'rectangle':
        P.align = None
        P.angle_inc_E_field = UP.angle_inc_E_field
        P.length_BZ_ortho = UP.length_BZ_ortho     # Size of the Brillouin zone in atomic units
        P.length_BZ_E_dir = UP.length_BZ_E_dir     # -"-
    else:
        sys.exit("BZ_type needs to be either hexagon or rectangle.")

    P.save_latex_pdf = False              # Save data as human readable PDF file
    if hasattr(UP, 'save_latex_pdf'):
        P.save_latex_pdf = UP.save_latex_pdf

    # Filename tail
    P.tail = 'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}_solver_{:s}_dk_order{}'\
        .format(P.E0_MVpcm, P.f_THz, P.alpha_fs, P.gauge, P.t0_fs, P.dt_fs, P.Nk1, P.Nk2, P.T1_fs, P.T2_fs, P.chirp_THz, P.phase, P.solver_method, P.dk_order)

    return P
