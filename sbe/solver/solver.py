from math import modf
import numpy as np
from numpy.fft import fft, fftfreq, fftshift, ifftshift
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode
from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.solver import make_current_path, make_polarization_path, make_emission_exact_path
from sbe.solver import make_electric_field


def sbe_solver(sys, dipole, params, curvature, electric_field_function=None):
    """
        Solver for the semiconductor bloch equation ( eq. (39) or (47) in https://arxiv.org/abs/2008.03177)
        for a two band system with analytical calculation of the dipole elements

        Author:
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters
        ----------
        sys : class
            Symbolic Hamiltonian of the system
        dipole : class
            Symbolic expression for the dipole elements (eq. (37/38))
        params :
            Parameters from the params.py file
        curvature : class
            Symbolic berry curvature (d(Ax)/d(ky) - d(Ay)/d(kx)) with
            A as the Berry connection (eq. (38))
        electric_field_function : function
            Jitted function of a user provided electric field.
            Can only take time as parameter. If is None takes
            electric field from sbe/solver/fields.py

        Returns
        -------
        params
        ------
        saves parameters of the calculation

        Iexact (file, 8 components)
        ------
        t : np.ndarray
            Nt-dimensional array of time-points
        I_exact_E_dir : np.ndarray
            Nt-dimensional array of current (eq. (59/64)) in E-field direction
        I_exact_ortho : np.ndarray
            Nt-dimensional array of current (eq. (59/64)) orthogonal to E-field
        freq/w : np.ndarray
            Nt-dimensional array of time-points in frequency domain
        Iw_exact_E_dir : np.ndarray
            Nt-dimensional array of fourier trafo of current in E-field direction
        Iw_exact_ortho : np.ndarray
            Nt-dimensional array of fourier trafo of current orthogonal to E-field
        Int_exact_E_dir : np.ndarray
            Nt-dimensional array of emission intensity (eq. (51)) in E-field direction
        Int_exact_ortho : np.ndarray
            Nt-dimensional array of emission intensity (eq. (51)) orthogonal to E-field

        Iapprox  (file, 8 components)
        -------
        approximate solutions, but same components as Iexact
    """
    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_full = params.save_full
    save_approx = params.save_approx
    save_txt = params.save_txt
    do_semicl = params.do_semicl
    gauge = params.gauge

    # System parameters
    a = params.a                                   # Lattice spacing
    e_fermi = params.e_fermi*co.eV_to_au           # Fermi energy
    temperature = params.temperature*co.eV_to_au   # Temperature

    # Driving field parameters
    E0 = params.E0*co.MVpcm_to_au                  # Driving pulse field amplitude
    w = params.w*co.THz_to_au                      # Driving pulse frequency
    chirp = params.chirp*co.THz_to_au              # Pulse chirp frequency
    alpha = params.alpha*co.fs_to_au               # Gaussian pulse width
    phase = params.phase                           # Carrier-envelope phase

    # Time scales
    T1 = params.T1*co.fs_to_au                     # Occupation damping time
    T2 = params.T2*co.fs_to_au                     # Polarization damping time
    gamma1 = 1/T1                                  # Occupation damping parameter
    gamma2 = 1/T2                                  # Polarization damping


    Nf = int((abs(2*params.t0))/params.dt)
    if modf((2*params.t0/params.dt))[0] > 1e-12:
        print("WARNING: The time window divided by dt is not an integer.")
    # Define a proper time window if Nt exists
    # +1 assures the inclusion of tf in the calculation
    Nt = Nf + 1
    t0 = params.t0*co.fs_to_au
    tf = -t0
    dt = params.dt*co.fs_to_au

    # Brillouin zone type
    BZ_type = params.BZ_type                       # Type of Brillouin zone

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1 = params.Nk1                           # kpoints in b1 direction
        Nk2 = params.Nk2                           # kpoints in b2 direction
        Nk = Nk1*Nk2                               # Total number of kpoints
        align = params.align                       # E-field alignment
        angle_inc_E_field = None
    elif BZ_type == '2line':
        align = None
        angle_inc_E_field = params.angle_inc_E_field
        Nk1 = params.Nk1
        Nk2 = params.Nk2
        Nk = Nk1*Nk2

    b1 = params.b1                                 # Reciprocal lattice vectors
    b2 = params.b2

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print_user_info(BZ_type, do_semicl, Nk, align, angle_inc_E_field, E0, w, alpha,
                        chirp, T2, tf-t0, dt)
    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        _kpnts, paths, area = hex_mesh(Nk1, Nk2, a, b1, b2, align)
        kweight = area/Nk
        dk = 1/Nk1
        if align == 'K':
            E_dir = np.array([1, 0])
        elif align == 'M':
            E_dir = np.array([np.cos(np.radians(-30)),
                              np.sin(np.radians(-30))])
        # BZ_plot(_kpnts, a, b1, b2, paths)
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                          np.sin(np.radians(angle_inc_E_field))])
        dk, kweight, _kpnts, paths = rect_mesh(params, E_dir)
        # BZ_plot(_kpnts, a, b1, b2, paths)

    # Time array construction flag
    t_constructed = False

    # Initialize electric_field, create fnumba and initialize ode solver
    if electric_field_function is None:
        electric_field = make_electric_field(E0, w, alpha, chirp, phase)
    else:
        electric_field = electric_field_function

    fnumba = make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field,
                         gauge=gauge, do_semicl=do_semicl)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method='bdf', max_step=dt)

    t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, P_E_dir, P_ortho, _dummy =\
        solution_containers(Nk1, Nk2, Nt, save_approx, save_full)

    # Exact emission function
    # Set after first run
    emission_exact_path = None
    # Approximate (kira & koch) emission function
    # Set after first run if save_approx=True
    current_path = None
    polarization_path = None

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    for Nk2_idx, path in enumerate(paths):
        print("Path: ", Nk2_idx + 1)
        if not save_full:
            # If we don't need the full solution only operate on path idx 0
            Nk2_idx = 0

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        if do_semicl:
            zeros = np.zeros(np.size(kx_in_path), dtype=np.complex)
            dipole_in_path = zeros
            A_in_path = zeros
        else:
            # Calculate the dipole components along the path
            di_00x = dipole.Axfjit[0][0](kx=kx_in_path, ky=ky_in_path)
            di_01x = dipole.Axfjit[0][1](kx=kx_in_path, ky=ky_in_path)
            di_11x = dipole.Axfjit[1][1](kx=kx_in_path, ky=ky_in_path)
            di_00y = dipole.Ayfjit[0][0](kx=kx_in_path, ky=ky_in_path)
            di_01y = dipole.Ayfjit[0][1](kx=kx_in_path, ky=ky_in_path)
            di_11y = dipole.Ayfjit[1][1](kx=kx_in_path, ky=ky_in_path)

            # Calculate the dot products E_dir.d_nm(k).
            # To be multiplied by E-field magnitude later.
            # A[0, 1, :] means 0-1 offdiagonal element
            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        ev = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)
        ecv_in_path = ec - ev

        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = initial_condition(e_fermi, temperature, ev, ec)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0, t0)\
            .set_f_params(path, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

        # Propagate through time
        # Index of current integration time step
        ti = 0

        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti % (Nt//20) == 0 and user_out):
                print('{:5.2f}%'.format((ti/Nt)*100))

            # Save solution each output step
            # Do not append the last element (A_field)
            # If save_full is False Nk2_idx is 0 as only the current path
            # is saved
            solution[:, Nk2_idx, ti, :] = solver.y[:-1].reshape(Nk1, 4)
            # Construct time array only once
            if not t_constructed:
                # Construct time and A_field only in first round
                t[ti] = solver.t
                A_field[ti] = solver.y[-1].real
                E_field[ti] = electric_field(t[ti])

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Increment time counter
            ti += 1

        if not t_constructed:
            # Construct the function after the first full run!
            emission_exact_path = make_emission_exact_path(sys, Nk1, Nt, E_dir, A_field,
                                                           gauge, do_semicl, curvature, E_field)
            if save_approx:
                # Only need kira & koch formulas if save_approx is set
                current_path = make_current_path(sys, Nk1, Nt, E_dir, A_field, gauge)
                polarization_path = make_polarization_path(dipole, Nk1, Nt, E_dir, A_field,
                                                           gauge)

        # Compute per path observables
        emission_exact_path(path, solution[:, Nk2_idx, :, :], I_exact_E_dir, I_exact_ortho)

        if save_approx:
            # Only calculate kira & koch formula if save_approx is set
            fv = solution[:, Nk2_idx, :, 0]
            fc = solution[:, Nk2_idx, :, 3]
            pcv = solution[:, Nk2_idx, :, 2]
            current_path(path, fv, fc, J_E_dir, J_ortho)
            polarization_path(path, pcv, P_E_dir, P_ortho)

        # Flag that time array has been built up
        t_constructed = True

    # Write solutions
    # Filename tail
    tail = 'E_{:.2f}_w_{:.2f}_a_{:.2f}_{}_t0_{:.2f}_NK1-{}_NK2-{}_T1_{:.2f}_T2_{:.2f}_chirp_{:.3f}_ph_{:.2f}'\
        .format(E0*co.au_to_MVpcm, w*co.au_to_THz, alpha*co.au_to_fs, gauge, params.t0, Nk1, Nk2, T1*co.au_to_fs, T2*co.au_to_fs, chirp*co.au_to_THz, phase)

    write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian(t, alpha), save_approx, save_txt)

    # Save the parameters of the calculation
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(params.__dict__))

    if save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution=solution, paths=paths,
                 electric_field=electric_field(t), A_field=A_field)


def make_fnumba(sys, dipole, E_dir, gamma1, gamma2, electric_field, gauge,
                do_semicl):
    """
        Initialization of the solver for the sbe ( eq. (39/47/80) in https://arxiv.org/abs/2008.03177)

        Author:
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters
        ----------
        sys : class
            Symbolic Hamiltonian of the system
        dipole : class
            Symbolic expression for the dipole elements (eq. (37/38))
        E_dir : np.ndarray
            2-dimensional array with the x and y component of the electric field
        gamma1 : float
            inverse of occupation damping time (T_1 in (eq. (?))
        gamma2 : float
            inverse of polarization damping time (T_2 in eq. (80))
        electric_field : jitted function
            absolute value of the instantaneous driving field E(t) (eq. (75))
        gauge: 'length' or 'velocity'
            parameter to determine which gauge is used in the routine
        do_semicl: boolean
            parameter to determine whether a semiclassical calculation will be done

        Returns
        -------
        f :
            right hand side of ode d/dt(rho(t)) = f(rho, t) (eq. (39/47/80))
    """
    ########################################
    # Wire the energies
    ########################################
    evf = sys.efjit[0]
    ecf = sys.efjit[1]

    ########################################
    # Wire the dipoles
    ########################################
    # kx-parameter
    di_00xf = dipole.Axfjit[0][0]
    di_01xf = dipole.Axfjit[0][1]
    di_11xf = dipole.Axfjit[1][1]

    # ky-parameter
    di_00yf = dipole.Ayfjit[0][0]
    di_01yf = dipole.Ayfjit[0][1]
    di_11yf = dipole.Ayfjit[1][1]

    @njit
    def flength(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

        # Gradient term coefficient
        electric_f = electric_field(t)
        D = electric_f/(2*dk)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            if k == 0:
                m = 4*(k+1)
                n = 4*(Nk_path-1)
            elif k == Nk_path-1:
                m = 0
                n = 4*(k-1)
            else:
                m = 4*(k+1)
                n = 4*(k-1)

            # Energy gap e_2(k) - e_1(k) >= 0 at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = q*d_12(k)*E(t)
            # Rabi frequency conjugate: w_R_c = q*d_21(k)*E(t)
            wr = dipole_in_path[k]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = q*(d_11(k) - d_22(k))*E(t)
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c

#            x[i] = 2*(wr*y[i+1]).imag + D*(y[m] - y[n]) \
#                - gamma1*(y[i]-y0[i])
#
#            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
#                - 1j*wr_c*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])
#
#            x[i+2] = x[i+1].conjugate()
#
#            x[i+3] = -2*(wr*y[i+1]).imag + D*(y[m+3] - y[n+3]) \
#                - gamma1*(y[i+3]-y0[i+3])

            # New solver routine (correct)
            x[i] = 2*(y[i+1]*wr_c).imag + D*(y[m] - y[n]) \
                   - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                     - 1j*wr*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag + D*(y[m+3] - y[n+3]) \
                     - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    @njit
    def fvelocity(t, y, kpath, _dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        k_shift = y[-1].real
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if do_semicl:
            zeros = np.zeros(kx.size, dtype=np.complex128)
            dipole_in_path = zeros
            A_in_path = zeros
        else:
            di_00x = di_00xf(kx=kx, ky=ky)
            di_01x = di_01xf(kx=kx, ky=ky)
            di_11x = di_11xf(kx=kx, ky=ky)
            di_00y = di_00yf(kx=kx, ky=ky)
            di_01y = di_01yf(kx=kx, ky=ky)
            di_11y = di_11yf(kx=kx, ky=ky)

            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.complex128)

        # Gradient term coefficient
        electric_f = electric_field(t)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
#            x[i] = 2*(wr*y[i+1]).imag - gamma1*(y[i]-y0[i])
#
#            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr_c*(y[i]-y[i+3])
#
#            x[i+2] = x[i+1].conjugate()
#
#            x[i+3] = -2*(wr*y[i+1]).imag - gamma1*(y[i+3]-y0[i+3])

            x[i] = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f

        return x

    freturn = None
    if gauge == 'length':
        print("Using length gauge")
        freturn = flength
    elif gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")

    # The python solver does not directly accept jitted functions so we wrap it
    def f(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0):
        return freturn(t, y, kpath, dk, ecv_in_path, dipole_in_path, A_in_path, y0)

    return f

def solution_containers(Nk1, Nk2, Nt, save_approx, save_full, zeeman=False):
    """
        Function that builds the containers on which the solutions of the SBE, 
        as well as the currents will be written
    """
    # Solution containers
    t = np.empty(Nt)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    if save_full:
        # Make container for full solution if it is needed
        solution = np.empty((Nk1, Nk2, Nt, 4), dtype=complex)
    else:
        # Only one path needed at a time if no full solution is needed
        solution = np.empty((Nk1, 1, Nt, 4), dtype=complex)

    A_field = np.empty(Nt, dtype=np.float64)
    E_field = np.empty(Nt, dtype=np.float64)

    I_exact_E_dir = np.zeros(Nt, dtype=np.float64)
    I_exact_ortho = np.zeros(Nt, dtype=np.float64)

    if save_approx:
        J_E_dir = np.zeros(Nt, dtype=np.float64)
        J_ortho = np.zeros(Nt, dtype=np.float64)
        P_E_dir = np.zeros(Nt, dtype=np.float64)
        P_ortho = np.zeros(Nt, dtype=np.float64)
    else:
        J_E_dir = None
        J_ortho = None
        P_E_dir = None
        P_ortho = None

    if zeeman:
        Zee_field = np.empty((Nt, 3), dtype=np.float64)
        return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
            P_E_dir, P_ortho, Zee_field

    return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
        P_E_dir, P_ortho, None


def initial_condition(e_fermi, temperature, ev, ec):
    '''
    Occupy conduction band according to inital Fermi energy and temperature
    '''
    knum = ec.size
    zeros = np.zeros(knum, dtype=np.float64)
    distrib_ec = np.zeros(knum, dtype=np.float64)
    distrib_ev = np.zeros(knum, dtype=np.float64)
    if temperature > 1e-5:
        distrib_ec += 1/(np.exp((ec-e_fermi)/temperature) + 1)
        distrib_ev += 1/(np.exp((ev-e_fermi)/temperature) + 1)
        return np.array([distrib_ev, zeros, zeros, distrib_ec]).flatten('F')

    smaller_e_fermi_ev = (e_fermi - ev) > 0
    smaller_e_fermi_ec = (e_fermi - ec) > 0

    distrib_ev[smaller_e_fermi_ev] += 1
    distrib_ec[smaller_e_fermi_ec] += 1
    return np.array([distrib_ev, zeros, zeros, distrib_ec]).flatten('F')


def diff(x, y):
    '''
    Takes the derivative of y w.r.t. x
    '''
    if len(x) != len(y):
        raise ValueError('Vectors have different lengths')
    if len(y) == 1:
        return 0

#    dx = np.gradient(x)
#    dy = np.gradient(y)
    dx = np.roll(x,-1) - np.roll(x,1)
    dy = np.roll(y,-1) - np.roll(y,1)

    return dy/dx

def fourier(dt, data):
    '''
    Calculate the phase correct fourier transform with proper normalization
    for calculations centered around t=0
    '''
    return (dt/np.sqrt(2*np.pi))*fftshift(fft(ifftshift(data)))



def gaussian(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    # sigma = sqrt(2)*alpha
    # # 1/(2*np.sqrt(np.pi)*alpha)*np.exp(-t**2/(2*alpha)**2)
    return np.exp(-t**2/(2*alpha)**2)


def write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian_envelope, save_approx, save_txt):
    """
        Calculates the Emission Intensity I(omega) (eq. 51 in https://arxiv.org/abs/2008.03177)

        Author: 
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters
        ----------

        tail : str
        kweight : float
        w : float
            driving pulse frequency
        t : ndarray
            array of the time points corresponding to a solution of the sbe
        I_exact_E_dir: ndarray
            exact emission j(t) in E-field direction
        I_exact_ortho : ndarray
            exact emission j(t) orthogonal to E-field
        J_E_dir : ndarray
            approximate emission j(t) in E-field direction
        J_E_ortho : ndarray
            approximate emission j(t) orthogonal to E-field
        P_E_dir : ndarray
            polarization E-field direction
        P_E_ortho : ndarray 
            polarization orthogonal to E-field
        gaussian_envelope : function
            gaussian function to multiply to a function before Fourier transform
        save_approx : boolean
            determines whether approximate solutions should be saved
        save_txt : boolean
            determines whether a .txt file with the soluion should be saved

        Returns:
        --------

        savefiles (see documentation of sbe_solver())
    """
    # Fourier transforms
    # 1/(3c^3) in atomic units
    prefac_emission = 1/(3*(137.036**3))
    dt_out = t[1] - t[0]
    freq = fftshift(fftfreq(t.size, d=dt_out))
    if save_approx:
        # Only do approximate emission fourier transforms if save_approx is set
        I_E_dir = kweight*(diff(t, P_E_dir) + J_E_dir)
        I_ortho = kweight*(diff(t, P_ortho) + J_ortho)

        I_intra_E_dir = J_E_dir*kweight
        I_intra_ortho = J_ortho*kweight

        I_inter_E_dir = diff(t, P_E_dir)*kweight
        I_inter_ortho = diff(t, P_ortho)*kweight

        Iw_E_dir = fourier(dt_out, I_E_dir*gaussian_envelope)
        Iw_ortho = fourier(dt_out, I_ortho*gaussian_envelope)

        Iw_intra_E_dir = fourier(dt_out, I_intra_E_dir*gaussian_envelope)
        Iw_intra_ortho = fourier(dt_out, I_intra_ortho*gaussian_envelope)

        Iw_inter_E_dir = fourier(dt_out, I_inter_E_dir*gaussian_envelope)
        Iw_inter_ortho = fourier(dt_out, I_inter_ortho*gaussian_envelope)

        # Approximate Emission intensity
        Int_E_dir = prefac_emission*(freq**2)*np.abs(Iw_E_dir)**2
        Int_ortho = prefac_emission*(freq**2)*np.abs(Iw_ortho)**2

        Int_intra_E_dir = prefac_emission*(freq**2)*np.abs(Iw_intra_E_dir)**2
        Int_intra_ortho = prefac_emission*(freq**2)*np.abs(Iw_intra_ortho)**2

        Int_inter_E_dir = prefac_emission*(freq**2)*np.abs(Iw_inter_E_dir)**2
        Int_inter_ortho = prefac_emission*(freq**2)*np.abs(Iw_inter_ortho)**2

        I_approx_name = 'Iapprox_' + tail

        np.save(I_approx_name, [t, I_E_dir, I_ortho,
                                freq/w, Iw_E_dir, Iw_ortho,
                                Int_E_dir, Int_ortho,
                                I_intra_E_dir, I_intra_ortho,
                                Int_intra_E_dir, Int_intra_ortho,
                                I_inter_E_dir, I_inter_ortho,
                                Int_inter_E_dir, Int_inter_ortho])

        if save_txt:
            np.savetxt(I_approx_name + '.dat',
                       np.column_stack([t.real, I_E_dir.real, I_ortho.real,
                                        (freq/w).real, Iw_E_dir.real, Iw_E_dir.imag, Iw_ortho.real, Iw_ortho.imag,
                                        Int_E_dir.real, Int_ortho.real]),
                       header="t, I_E_dir, I_ortho, freqw/w, Re(Iw_E_dir), Im(Iw_E_dir), Re(Iw_ortho), Im(Iw_ortho), Int_E_dir, Int_ortho",
                       fmt='%+.34f')

    ##############################################################
    # Always calculate exact emission formula
    ##############################################################
    # kweight is different for 2line and full
    I_exact_E_dir *= kweight
    I_exact_ortho *= kweight

    Iw_exact_E_dir = fourier(dt_out, I_exact_E_dir*gaussian_envelope)
    Iw_exact_ortho = fourier(dt_out, I_exact_ortho*gaussian_envelope)
    Int_exact_E_dir = prefac_emission*(freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = prefac_emission*(freq**2)*np.abs(Iw_exact_ortho)**2

    I_exact_name = 'Iexact_' + tail
    np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho,
                           freq/w, Iw_exact_E_dir, Iw_exact_ortho,
                           Int_exact_E_dir, Int_exact_ortho])
    if save_txt:
        np.savetxt(I_exact_name + '.dat',
                   np.column_stack([t.real, I_exact_E_dir.real, I_exact_ortho.real,
                                    (freq/w).real, Iw_exact_E_dir.real, Iw_exact_E_dir.imag, Iw_exact_ortho.real, Iw_exact_ortho.imag,
                                    Int_exact_E_dir.real, Int_exact_ortho.real]),
                   header="t, I_exact_E_dir, I_exact_ortho, freqw/w, Re(Iw_exact_E_dir), Im(Iw_exact_E_dir), Re(Iw_exact_ortho), Im(Iw_exact_ortho), Int_exact_E_dir, Int_exact_ortho",
                   fmt='%+.34f')


def print_user_info(BZ_type, do_semicl, Nk, align, angle_inc_E_field, E0, w, alpha, chirp,
                    T2, tfmt0, dt, B0=None, mu=None, incident_angle=None):
    """
        Function that prints the input parameters if usr_info = True
    """
    print("Input parameters:")
    print("Brillouin zone:                 " + BZ_type)
    print("Do Semiclassics                 " + str(do_semicl))
    print("Number of k-points              = " + str(Nk))
    if BZ_type == 'full':
        print("Driving field alignment         = " + align)
    elif BZ_type == '2line':
        print("Driving field direction         = " + str(angle_inc_E_field))
    if B0 is not None:
        print("Incident angle                  = " + str(np.rad2deg(incident_angle)))
    print("Driving amplitude (MV/cm)[a.u.] = " + "("
          + '{:.6f}'.format(E0*co.au_to_MVpcm) + ")"
          + "[" + '{:.6f}'.format(E0) + "]")
    if B0 is not None:
        print("Magnetic amplitude (T)[a.u.]    = " + "("
              + '%.6f'%(B0*co.au_to_T) + ")"
              + "[" + '%.6f'%(B0) + "]")
        print("Magnetic moments ", mu)
    print("Pulse Frequency (THz)[a.u.]     = " + "("
          + '{:.6f}'.format(w*co.au_to_THz) + ")"
          + "[" + '{:.6f}'.format(w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "("
          + '{:.6f}'.format(alpha*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(alpha) + "]")
    print("Chirp rate (THz)[a.u.]          = " + "("
          + '{:.6f}'.format(chirp*co.au_to_THz) + ")"
          + "[" + '{:.6f}'.format(chirp) + "]")
    print("Damping time (fs)[a.u.]         = " + "("
          + '{:.6f}'.format(T2*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(T2) + "]")
    print("Total time (fs)[a.u.]           = " + "("
          + '{:.6f}'.format(tfmt0*co.au_to_fs) + ")"
          + "[" + '{:.5f}'.format(tfmt0) + "]")
    print("Time step (fs)[a.u.]            = " + "("
          + '{:.6f}'.format(dt*co.au_to_fs) + ")"
          + "[" + '{:.6f}'.format(dt) + "]")


def BZ_plot(kpnts, a, b1, b2, paths, si_units=True):
    """
        Function that plots the brillouin zone
    """
    if si_units:
        a *= co.au_to_as
        kpnts *= co.as_to_au
        b1 *= co.as_to_au
        b2 *= co.as_to_au

    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = plt.figure(figsize=(10, 10))
    ax = BZ_fig.add_subplot(111, aspect='equal')

    for b in ((0, 0), b1, -b1, b2, -b2, b1+b2, -b1-b2):
        poly = RegularPolygon(b, 6, radius=R, orientation=np.pi/6, fill=False)
        ax.add_patch(poly)

#    ax.arrow(-0.5*E_dir[0], -0.5*E_dir[1], E_dir[0], E_dir[1],
#             width=0.005, alpha=0.5, label='E-field')

    plt.scatter(0, 0, s=15, c='black')
    plt.text(0.01, 0.01, r'$\Gamma$')
    plt.scatter(r*np.cos(-np.pi/6), r*np.sin(-np.pi/6), s=15, c='black')
    plt.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, r'$M$')
    plt.scatter(R, 0, s=15, c='black')
    plt.text(R, 0.02, r'$K$')
    plt.scatter(kpnts[:, 0], kpnts[:, 1], s=10)
    plt.xlim(-7.0/a, 7.0/a)
    plt.ylim(-7.0/a, 7.0/a)

    if si_units:
        plt.xlabel(r'$k_x \text{ in } 1/\si{\angstrom}$')
        plt.ylabel(r'$k_y \text{ in } 1/\si{\angstrom}$')
    else:
        plt.xlabel(r'$k_x \text{ in } 1/a_0$')
        plt.ylabel(r'$k_y \text{ in } 1/a_0$')

    for path in paths:
        if si_units:
            plt.plot(co.as_to_au*path[:, 0], co.as_to_au*path[:, 1])
        else:
            plt.plot(path[:, 0], path[:, 1])

    plt.show()
