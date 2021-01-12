import time
from math import ceil, modf
import numpy as np
from numpy.fft import fft, fftfreq, fftshift, ifftshift
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

import sbe.example
from sbe.brillouin import hex_mesh, rect_mesh
from sbe.utility import conversion_factors as co
from sbe.solver import make_electric_field
from sbe.solver import make_matrix_elements_hderiv, make_matrix_elements_dipoles, current_per_path
from sbe.dipole import diagonalize, dipole_elements

def sbe_solver_n_bands(params, sys, dipole, curvature):
    """
        Solver for the semiconductor bloch equation ( eq. (39) or (47) in https://arxiv.org/abs/2008.03177)
        for a n band system with numerical calculation of the dipole elements (unfinished - analytical dipoles
        can be used for n=2)

        Author: Adrian Seith (adrian.seith@ur.de)
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
    # Start time of sbe_solver
    start_time = time.perf_counter()

    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_full = params.save_full
    save_approx = params.save_approx
    save_txt = params.save_txt
    do_semicl = params.do_semicl
    gauge = params.gauge

    if hasattr(params, 'solver_method'):           # 'adams' non-stiff and 'bdf' stiff problems
        method = params.solver_method
    else:
        method = 'bdf'

    # higher precision (quadruple for reducing numerical noise
    precision = 'default'
    if hasattr(params, 'precision'):
        precision = params.precision

    if precision == 'default':
        type_real_np    = np.float64
        type_complex_np = np.complex128
    elif precision == 'quadruple':
        type_real_np    = np.float128
        type_complex_np = np.complex256
        if method != 'rk4': quit("Error: Quadruple precision only works with Runge-Kutta 4 ODE solver.")
    else: quit("Only default or quadruple precision available.")

    dk_order = 8
    if hasattr(params, 'dk_order'):                # Accuracy order of numerical density-matrix k-deriv.
        dk_order = params.dk_order                 # when using the length gauge (avail: 2,4,6,8)
        if dk_order not in [2, 4, 6, 8]:
            quit("dk_order needs to be either 2, 4, 6, or 8.")

    # System parameters
    n = params.n
    gidx = params.gidx
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
        dk, kweight, _kpnts, paths = rect_mesh(params, E_dir, type_real_np)
        # BZ_plot(_kpnts, a, b1, b2, paths)

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # Time array construction flag
    t_constructed = False

    # Initialize electric_field, create fnumba and initialize ode solver
    electric_field = make_electric_field(E0, w, alpha, chirp, phase, type_real_np)

    fnumba = make_fnumba(n, E_dir, gamma1, gamma2, electric_field, params, dk, dk_order, gauge=gauge)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method=method, max_step=dt)

    t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, P_E_dir, P_ortho =\
        solution_containers(Nk1, Nk2, Nt, params.n, save_approx, save_full)

    # Exact emission function
    # Set after first run
    emission_exact_path = None
    # Approximate (kira & koch) emission function
    # Set after first run if save_approx=True
    current_path = None
    polarization_path = None

    emission_exact_E_dir = np.zeros(Nt, dtype=np.complex128)
    emission_intraband_E_dir = np.zeros(Nt, dtype=np.complex128)
    emission_exact_ortho = np.zeros(Nt, dtype=np.complex128)
    emission_intraband_ortho = np.zeros(Nt, dtype=np.complex128)

    dipole_in_path = np.empty([Nk1, n, n], dtype=np.complex128)
    dipole_in_path2 = np.empty([Nk1, n, n], dtype=np.complex128)
    dipole_ortho = np.empty([Nk1, n, n], dtype=np.complex128)
    e_in_path = np.empty([Nk1, n], dtype=np.complex128)  
    e_in_path2 = np.empty([Nk1, n], dtype=np.complex128) 

    hnp = sys.numpy_hamiltonian()
    e, wf = diagonalize(params, hnp, paths)  

    if params.dipole_numerics:
    # Calculate the dipole elements on the full k-mesh
        if user_out: 
            print("Calculating dipoles...")
              
        dipole_x, dipole_y = dipole_elements(params, hnp, paths)

    # Only define full density matrix solution if save_full is True
    if save_full:
        solution_full = np.empty((Nk1, Nk2, Nt, 4), dtype=np.complex128)

    ###########################################################################
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone

    for Nk2_idx, path in enumerate(paths):
        print("Solving SBE for Path: ", Nk2_idx + 1)

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

#############################################################################
        if params.dipole_numerics:    
        # Evaluate the dipole components along the path

            # Calculate the dot products E_dir.d_nm(k).
            # To be multiplied by E-field magnitude later.            
            dipole_in_path = (E_dir[0]*dipole_x[:, Nk2_idx, :, :] + E_dir[1]*dipole_y[:, Nk2_idx, :, :])
            dipole_ortho = (E_ort[0]*dipole_x[:, Nk2_idx, :, :] + E_ort[1]*dipole_y[:, Nk2_idx, :, :])
            e_in_path = e[:, Nk2_idx, :]

#############################################################################
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
            dipole_in_path[:, 0, 1] = E_dir[0]*di_01x + E_dir[1]*di_01y
            dipole_in_path[:, 1, 0] = dipole_in_path[:, 0, 1].conjugate()
            dipole_in_path[:, 0, 0] = E_dir[0]*di_00x + E_dir[1]*di_00y
            dipole_in_path[:, 1, 1] = E_dir[0]*di_11x + E_dir[1]*di_11y

            dipole_ortho[:, 0, 1] = E_ort[0]*di_01x + E_ort[1]*di_01y
            dipole_ortho[:, 1, 0] = dipole_in_path[:, 0, 1].conjugate()
            dipole_ortho[:, 0, 0] = E_ort[0]*di_00x + E_ort[1]*di_00y
            dipole_ortho[:, 1, 1] = E_ort[0]*di_11x + E_ort[1]*di_11y

            e_in_path[:, 0] = sys.efjit[0](kx=kx_in_path, ky=ky_in_path)
            e_in_path[:, 1] = sys.efjit[1](kx=kx_in_path, ky=ky_in_path)
#############################################################################


    # for i in range(n):
        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = initial_condition(e_fermi, temperature, e_in_path)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath

        solver.set_initial_value(y0, t0)\
            .set_f_params(dipole_in_path, e_in_path, y0)

        # Propagate through time

        # Index of current integration time step
        ti = 0

        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti % (Nt//20) == 0 and user_out):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Save solution each output step
            # Do not append the last element (A_field)
            # If save_full is False Nk2_idx is 0 as only the current path
            # is saved
            solution[:, ti, :, :] = solver.y[:-1].reshape(Nk1, n, n)

            # Only write full density matrix solution if save_full is True
            if save_full:
                solution_full[:, Nk2_idx, ti, :] = solution

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

        # Compute per path observables
        if user_out: 
            print("Calculating emission of the current path...")

        mel_in_path, mel_ortho = make_matrix_elements_dipoles(params, hnp, paths, dipole_in_path, dipole_ortho, e_in_path, E_dir, Nk2_idx)
        #mel_in_path, mel_ortho = make_matrix_elements_hderiv(params, hnp, paths, wf, E_dir, Nk2_idx)     # NOT ACCURATE (YET!) 

        current_in_path, current_in_path_intraband, current_ortho, current_ortho_intraband = current_per_path(params, Nt, mel_in_path, mel_ortho, solution)
        
        emission_exact_E_dir += current_in_path
        emission_intraband_E_dir += current_in_path_intraband
        emission_exact_ortho += current_ortho
        emission_intraband_ortho += current_ortho_intraband
        
        # Flag that time array has been built up
        t_constructed = True
    I_exact_E_dir = emission_exact_E_dir
    I_exact_orth = emission_exact_ortho
    J_E_dir = emission_intraband_E_dir
    J_ortho = emission_intraband_ortho

    # End time of solver loop
    end_time = time.perf_counter()

    # Write solutions
    # Filename tail
    tail = 'E_{:.4f}_w_{:.1f}_a_{:.1f}_{}_t0_{:.1f}_dt_{:.6f}_NK1-{}_NK2-{}_T1_{:.1f}_T2_{:.1f}_chirp_{:.3f}_ph_{:.2f}'\
        .format(E0*co.au_to_MVpcm, w*co.au_to_THz, alpha*co.au_to_fs, gauge, params.t0, params.dt, Nk1, Nk2, T1*co.au_to_fs, T2*co.au_to_fs, chirp*co.au_to_THz, phase)

    write_current_emission(tail, kweight, w, t, I_exact_E_dir, I_exact_ortho,
                           J_E_dir, J_ortho, P_E_dir, P_ortho,
                           gaussian(t, alpha), save_approx, save_txt)

    # Save the parameters of the calculation
    run_time = end_time - start_time
    params_name = 'params_' + tail + '.txt'
    paramsfile = open(params_name, 'w')
    paramsfile.write(str(params.__dict__) + "\n\n")
    paramsfile.write("Runtime: {:.16f} s".format(run_time))
    paramsfile.close()

    if save_full:
        S_name = 'Sol_' + tail
        np.savez(S_name, t=t, solution_full=solution_full, paths=paths,
                 electric_field=electric_field(t), A_field=A_field)

def make_fnumba(n, E_dir, gamma1, gamma2, electric_field, params, dk, dk_order, gauge):
    """
        Initialization of the solver for the SBE ( eq. (39/40(80) in https://arxiv.org/abs/2008.03177)
        
        Author: Adrian Seith (adrian.seith@ur.de)
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)

        Parameters:
        ----------- 
            n : integer 
                Number of bands of the system
            E_dir : np.ndarray
                x and y component of the direction of the electric field
            gamma1 : float
                occupation damping parameter
            gamma2 : float
                polarization damping
            electric_field : function
                returns the instantaneous driving field
            gauge : str
                length or velocity gauge (only v. implemented)
        
        Returns:
        --------
            freturn : function that is the right hand side of the ode
    """
    if gauge == 'length':
        print('Using length gauge')
    elif gauge == 'velocity':
        print('Using velocity gauge')
    else:
        raise AttributeError("You have to either assign velocity or length gauge")
    
    Nk1 = params.Nk1

    @njit
    def fnumba(t, y, dipole_in_path, e_in_path, y0):
        """
            function that multiplies the block-structure of the matrices of the RHS
            of the SBE with the solution vector
        """
        # x != y(t+dt)
        x = np.zeros(np.shape(y), dtype=np.complex128)
        # Gradient term coefficient
        electric_f = electric_field(t)

        D = electric_f/dk

        for k in range(Nk1):
            right4 = (k+4)
            right3 = (k+3)
            right2 = (k+2)
            right  = (k+1)
            left   = (k-1)
            left2  = (k-2)
            left3  = (k-3)
            left4  = (k-4)
            if k == 0:
                left   = (Nk1-1)
                left2  = (Nk1-2)
                left3  = (Nk1-3)           
                left4  = (Nk1-4)           
            elif k == 1 and dk_order >= 4:
                left2  = (Nk1-1)
                left3  = (Nk1-2)            
                left4  = (Nk1-3)          
            elif k == 2 and dk_order >= 6:
                left3  = (Nk1-1)          
                left4  = (Nk1-2) 
            elif k == 3 and dk_order >= 8:
                left4  = (Nk1-1) 
            elif k == Nk1-1:
                right4 = 3
                right3 = 2
                right2 = 1
                right  = 0
            elif k == Nk1-2 and dk_order >= 4:
                right4 = 2
                right3 = 1
                right2 = 0
            elif k == Nk1-3 and dk_order >= 6:
                right4 = 1
                right3 = 0
            elif k == Nk1-4 and dk_order >= 8:
                right4 = 0
            for i in range(n):
                for j in range(n):
                    x[k*(n**2) + i*n + j] += -1j * ( e_in_path[k, i] - e_in_path[k, j] ) * y[k*(n**2) + i*n + j]
                    
                    if dk_order ==2:
                        x[k*(n**2) + i*n + j] += D * (  y[right*(n**2) + i*n + j]/2 - y[left*(n**2) + i*n + j]/2 )
                    elif dk_order ==4:
                        x[k*(n**2) + i*n + j] += D * (- y[right2*(n**2) + i*n + j]/12 + 2/3*y[right*(n**2) + i*n + j] \
                                                      - 2/3*y[left*(n**2) + i*n + j] + y[left2*(n**2) + i*n + j]/12 )
                    elif dk_order ==6:
                        x[k*(n**2) + i*n + j] += D * (  y[right3*(n**2) + i*n + j]/60 - 3/20*y[right2*(n**2) + i*n + j] + 3/4*y[right*(n**2) + i*n + j] \
                                                      - y[left3*(n**2) + i*n + j]/60 + 3/20*y[left2*(n**2) + i*n + j] - 3/4*y[left*(n**2) + i*n + j] )
                    elif dk_order ==8:
                        x[k*(n**2) + i*n + j] += D * (- y[right4*(n**2) + i*n + j]/280 + 4/105*y[right3*(n**2) + i*n + j] \
                                                      -  1/5*y[right2*(n**2) + i*n + j] + 4/5*y[right*(n**2) + i*n + j] \
                                                      + y[left4*(n**2) + i*n + j]/280 - 4/105*y[left3*(n**2) + i*n + j] \
                                                      + 1/5*y[left2*(n**2) + i*n + j] - 4/5*y[left*(n**2) + i*n + j] )

                    if i == j:
                        x[k*(n**2) + i*n + j] += - gamma1 * (y[k*(n**2) + i*n + j] - y0[k*(n**2) + i*n + j])
                    else: 
                        x[k*(n**2) + i*n + j] += - gamma2 * y[k*(n**2) + i*n + j]
                    for nbar in range(n):
                        if i == j:
                            x[k*(n**2) + i*n + j] += 2 * np.imag( electric_f * y[k*(n**2) + i*n + nbar] * dipole_in_path[k, nbar, i] )
                        else:
                            x[k*(n**2) + i*n + j] += -1j * electric_f * ( y[k*(n**2) + i*n + nbar] * dipole_in_path[k, nbar, j] - dipole_in_path[k, i, nbar] * y[k*(n**2) + nbar*n + j])

        x[-1] = -electric_f

        return x
    def f(t, y, dipole_in_path, e_in_path, y0):
        return fnumba(t, y, dipole_in_path, e_in_path, y0)

    return f

def solution_containers(Nk1, Nk2, Nt, n, save_approx, save_full, zeeman=False):
    """
        Function that builds the containers on which the solutions of the SBE, 
        as well as the currents will be written
    """
    # Solution containers
    t = np.empty(Nt)

    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e

    # Only one path needed at a time if no full solution is needed
    solution = np.empty((Nk1, Nt, n, n), dtype=np.complex128)

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
        Zee_field = np.empty((params.Nt, 3), dtype=np.float64)
        return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
            P_E_dir, P_ortho, Zee_field

    return t, A_field, E_field, solution, I_exact_E_dir, I_exact_ortho, J_E_dir, J_ortho, \
        P_E_dir, P_ortho


def initial_condition(e_fermi, temperature, e_in_path): # Check if this does what it should!
    '''
    Occupy conduction band according to inital Fermi energy and temperature
    '''
    num_kpoints = e_in_path[:, 0].size
    num_bands = e_in_path[0, :].size
    distrib_bands = np.zeros([num_kpoints, num_bands], dtype=np.complex128)
    initial_condition = np.zeros([num_kpoints, num_bands, num_bands], dtype=np.complex128)
    if temperature > 1e-5:
        distrib_bands += 1/(np.exp((e_in_path-e_fermi)/temperature) + 1)
    else:
        smaller_e_fermi = (e_fermi - e_in_path) > 0
        distrib_bands[smaller_e_fermi] += 1

    for k in range(num_kpoints):
        initial_condition[k, :, :] = np.diag(distrib_bands[k, :])
    return initial_condition.flatten('C')

def diff(x, y):
    '''
    Takes the derivative of y w.r.t. x
    '''
    if len(x) != len(y):
        raise ValueError('Vectors have different lengths')
    if len(y) == 1:
        return 0

    dx = np.roll(x,-1) - np.roll(x,1)
    dy = np.roll(y,-1) - np.roll(y,1)

    return dy/dx


def gaussian(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    # sigma = sqrt(2)*alpha
    # # 1/(2*np.sqrt(np.pi)*alpha)*np.exp(-t**2/(2*alpha)**2)
    return np.exp(-t**2/(2*alpha)**2)

def fourier(dt, data):
    '''
    Calculate the phase correct fourier transform with proper normalization
    for calculations centered around t=0
    '''
    return (dt/np.sqrt(2*np.pi))*fftshift(fft(ifftshift(data)))

def ifourier(dt, data):
    '''
    Calculate the phase correct inverse fourier transform with proper normalization
    for calculations centered around t=0
    '''
    return (np.sqrt(2*np.pi)/dt)*fftshift(ifft(ifftshift(data)))

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