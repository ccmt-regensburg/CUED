import numpy as np
import numpy.linalg as lin
from cued.utility import conditional_njit, evaluate_njit_matrix


def make_rhs_ode_2_band(sys, electric_field, P):
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
    gamma1 = P.gamma1
    gamma2 = P.gamma2
    type_complex_np = P.type_complex_np
    dk_order = P.dk_order
    dm_dynamics_method = P.dm_dynamics_method
    E_dir = P.E_dir
    gauge = P.gauge
    do_fock = P.do_fock
    Nk2 = P.Nk2
    split_paths = P.split_paths
    split_order = P.split_order

    if sys.system == 'ana':

        sys.make_eigensystem_dipole(P)

        ########################################
        # Wire the energies
        ########################################
        evf = sys.efjit[0]
        ecf = sys.efjit[1]

        ########################################
        # Wire the dipoles
        ########################################
        # kx-parameter
        di_00xf = sys.Axfjit[0][0]
        di_01xf = sys.Axfjit[0][1]
        di_11xf = sys.Axfjit[1][1]

        # ky-parameter
        di_00yf = sys.Ayfjit[0][0]
        di_01yf = sys.Ayfjit[0][1]
        di_11yf = sys.Ayfjit[1][1]

        # Coulomb-interaction matrix
        v_k_kprime = sys.v_k_kprime   
      
    @conditional_njit(P.type_complex_np)
    def flength(t, y, kpath, dipole_in_path, e_in_path, y0, dk, rho, Nk2_idx):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        # Gradient term coefficient
        electric_f = electric_field(t)
        D = electric_f/dk

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            right4 = 4*(k+4)
            right3 = 4*(k+3)
            right2 = 4*(k+2)
            right  = 4*(k+1)
            left   = 4*(k-1)
            left2  = 4*(k-2)
            left3  = 4*(k-3)
            left4  = 4*(k-4)
            if k == 0:
                left   = 4*(Nk_path-1)
                left2  = 4*(Nk_path-2)
                left3  = 4*(Nk_path-3)
                left4  = 4*(Nk_path-4)
            elif k == 1 and dk_order >= 4:
                left2  = 4*(Nk_path-1)
                left3  = 4*(Nk_path-2)
                left4  = 4*(Nk_path-3)
            elif k == 2 and dk_order >= 6:
                left3  = 4*(Nk_path-1)
                left4  = 4*(Nk_path-2)
            elif k == 3 and dk_order >= 8:
                left4  = 4*(Nk_path-1)
            elif k == Nk_path-1:
                right4 = 4*3
                right3 = 4*2
                right2 = 4*1
                right  = 4*0
            elif k == Nk_path-2 and dk_order >= 4:
                right4 = 4*2
                right3 = 4*1
                right2 = 4*0
            elif k == Nk_path-3 and dk_order >= 6:
                right4 = 4*1
                right3 = 4*0
            elif k == Nk_path-4 and dk_order >= 8:
                right4 = 4*0

            # Energy gap e_2(k) - e_1(k) >= 0 at point k
            ecv = e_in_path[k, 1] - e_in_path[k, 0]

            # Berry connection
            A_in_path = dipole_in_path[k, 0, 0] - dipole_in_path[k, 1, 1]

            # Rabi frequency: w_R = q*d_12(k)*E(t)
            # Rabi frequency conjugate: w_R_c = q*d_21(k)*E(t)
            wr = dipole_in_path[k, 0, 1]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = q*(d_11(k) - d_22(k))*E(t)
            wr_d_diag = A_in_path*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i]   = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

            # compute drift term via k-derivative
            if dk_order == 2:
                x[i]   += D*( y[right]/2   - y[left]/2  )
                x[i+1] += D*( y[right+1]/2 - y[left+1]/2 )
                x[i+3] += D*( y[right+3]/2 - y[left+3]/2 )
            elif dk_order == 4:
                x[i]   += D*(- y[right2]/12   + 2/3*y[right]   - 2/3*y[left]   + y[left2]/12 )
                x[i+1] += D*(- y[right2+1]/12 + 2/3*y[right+1] - 2/3*y[left+1] + y[left2+1]/12 )
                x[i+3] += D*(- y[right2+3]/12 + 2/3*y[right+3] - 2/3*y[left+3] + y[left2+3]/12 )
            elif dk_order == 6:
                x[i]   += D*(  y[right3]/60   - 3/20*y[right2]   + 3/4*y[right] \
                             - y[left3]/60    + 3/20*y[left2]    - 3/4*y[left] )
                x[i+1] += D*(  y[right3+1]/60 - 3/20*y[right2+1] + 3/4*y[right+1] \
                             - y[left3+1]/60  + 3/20*y[left2+1]  - 3/4*y[left+1] )
                x[i+3] += D*(  y[right3+3]/60 - 3/20*y[right2+3] + 3/4*y[right+3] \
                             - y[left3+3]/60  + 3/20*y[left2+3]  - 3/4*y[left+3] )
            elif dk_order == 8:
                x[i]   += D*(- y[right4]/280   + 4/105*y[right3]   - 1/5*y[right2]   + 4/5*y[right] \
                             + y[left4] /280   - 4/105*y[left3]    + 1/5*y[left2]    - 4/5*y[left] )
                x[i+1] += D*(- y[right4+1]/280 + 4/105*y[right3+1] - 1/5*y[right2+1] + 4/5*y[right+1] \
                             + y[left4+1] /280 - 4/105*y[left3+1]  + 1/5*y[left2+1]  - 4/5*y[left+1] )
                x[i+3] += D*(- y[right4+3]/280 + 4/105*y[right3+3] - 1/5*y[right2+3] + 4/5*y[right+3] \
                             + y[left4+3] /280 - 4/105*y[left3+3]  + 1/5*y[left2+3]  - 4/5*y[left+3] )

            # additional fock terms
            if do_fock:
                for kprime_y_idx in range(Nk2):
                    for kprime_x_idx in range(Nk_path):

                        kx_idx_old = k 
                        kprime_x_idx_old = kprime_x_idx
                        ky_idx_old = Nk2_idx
                        kprime_y_idx_old = kprime_y_idx

                        dist_kx_idx = int(np.abs(kx_idx_old - kprime_x_idx_old))
                        dist_ky_idx = int(np.abs(ky_idx_old - kprime_y_idx_old))

                        if dist_kx_idx != 0 or dist_ky_idx != 0: 

                            x[i]   += 2 * v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+2] * rho[kprime_x_idx, kprime_y_idx, 0, 1] ).imag

                            x[i+1] += 1j*v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+1] * (rho[kprime_x_idx, kprime_y_idx, 0, 0] - 1 - rho[kprime_x_idx, kprime_y_idx, 1, 1] ) \
                                                                    - ( y[i] -1 - y[i+3] ) * rho[kprime_x_idx, kprime_y_idx, 0, 1] )

                            x[i+3] += - 2 * v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+2] * rho[kprime_x_idx, kprime_y_idx, 0, 1] ).imag

        x[-1] = -electric_f
        return x

    @conditional_njit(P.type_complex_np)
    def pre_velocity(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if dm_dynamics_method == 'semiclassics':
            zero_arr = np.zeros(kx.size, dtype=type_complex_np)
            dipole_in_path = zero_arr
            A_in_path = zero_arr
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

        return ecv_in_path, dipole_in_path, A_in_path

    @conditional_njit(P.type_complex_np)
    def fvelocity(t, y, kpath, dipole_in_path, e_in_path, y0, dk, rho, Nk2_idx):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        ecv_in_path, dipole_in_path[:, 0, 1], A_in_path = pre_velocity(kpath, y[-1].real)
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        electric_f = electric_field(t)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        interaction_term = np.zeros((4*Nk_path))
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k, 0, 1]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

            # additional fock terms
            if do_fock:
                for kprime_y_idx in range(Nk2):
                    for kprime_x_idx in range(Nk_path):
                        if split_paths:
                            for o in range(split_order):
                                if Nk2_idx % split_order == o:
                                    kx_idx_old = k + o*Nk_path
                                    ky_idx_old = int( (Nk2_idx - o) / split_order )
                                if kprime_y_idx % split_order == o:
                                    kprime_x_idx_old = kprime_x_idx + o*Nk_path
                                    kprime_y_idx_old = int( (kprime_y_idx - o) / split_order )
                            
                        else:
                            kx_idx_old = k 
                            kprime_x_idx_old = kprime_x_idx
                            ky_idx_old = Nk2_idx
                            kprime_y_idx_old = kprime_y_idx

                        dist_kx_idx = int(np.abs(kx_idx_old - kprime_x_idx_old))
                        dist_ky_idx = int(np.abs(ky_idx_old - kprime_y_idx_old))

                        if dist_kx_idx != 0 or dist_ky_idx != 0: 

                            x[i]   += 2 * v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+2] * rho[kprime_x_idx, kprime_y_idx, 0, 1] ).imag

                            x[i+1] += 1j*v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+1] * (rho[kprime_x_idx, kprime_y_idx, 0, 0] - 1 - rho[kprime_x_idx, kprime_y_idx, 1, 1] ) \
                                                                    - ( y[i] -1 - y[i+3] ) * rho[kprime_x_idx, kprime_y_idx, 0, 1] )

                            x[i+3] += - 2 * v_k_kprime[dist_kx_idx, dist_ky_idx] * ( y[i+2] * rho[kprime_x_idx, kprime_y_idx, 0, 1] ).imag

            x[i+2] = x[i+1].conjugate()

        x[-1] = - electric_f

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
    def f(t, y, kpath, dipole_in_path, e_in_path, y0, dk, rho, Nk2_idx):
        return freturn(t, y, kpath, dipole_in_path, e_in_path, y0, dk, rho, Nk2_idx)

    return f

def make_rhs_ode_n_band(sys, electric_field, P):
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
    gamma1 = P.gamma1
    gamma2 = P.gamma2
    type_complex_np = P.type_complex_np
    type_real_np = P.type_real_np
    dk_order = P.dk_order
    gauge = P.gauge
    system = sys.system
    E_dir = P.E_dir
    n = P.n
    if system == 'ana' or system == 'num':
        for i in range(P.n):
            for j in range(P.n):
                globals()[f"hfjit_{i}{j}"] = sys.hfjit[i][j]
        degenerate_eigenvalues = sys.degenerate_eigenvalues

    if system == 'bandstructure' and gauge == 'velocity':
        if n != 2:
            raise AttributeError("Velocity gauge for custom bandstructures only works for 2 band systems")
        ########################################
        # Wire the energies
        ########################################
        evf = sys.efjit[0]
        ecf = sys.efjit[1]

        ########################################
        # Wire the dipoles
        ########################################
        # kx-parameter
        di_00xf = sys.dipole_xfjit[0][0]
        di_01xf = sys.dipole_xfjit[0][1]
        di_11xf = sys.dipole_xfjit[1][1]

        # ky-parameter
        di_00yf = sys.dipole_yfjit[0][0]
        di_01yf = sys.dipole_yfjit[0][1]
        di_11yf = sys.dipole_yfjit[1][1]       

    epsilon = P.epsilon
    gidx = P.gidx
    dm_dynamics_method = P.dm_dynamics_method

    @conditional_njit(type_complex_np)
    def fvelocity_custom_bs(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        ecv_in_path, dipole_in_path[:, 0, 1], A_in_path = pre_velocity_custom_bs(kpath, y[-1].real)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        electric_f = electric_field(t)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k, 0, 1]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
    
        return x

    @conditional_njit(P.type_complex_np)
    def pre_velocity_custom_bs(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if dm_dynamics_method == 'semiclassics':
            zero_arr = np.zeros(kx.size, dtype=type_complex_np)
            dipole_in_path = zero_arr
            A_in_path = zero_arr
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

        return ecv_in_path, dipole_in_path, A_in_path

    @conditional_njit(type_complex_np)
    def flength(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        """
            function that multiplies the block-structure of the matrices of the RHS
            of the SBE with the solution vector
        """
        # x != y(t+dt)
        x = np.zeros(np.shape(y), dtype=type_complex_np)
        # Gradient term coefficient
        electric_f = electric_field(t)

        D = electric_f/dk

        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            right4 = (k+4)
            right3 = (k+3)
            right2 = (k+2)
            right  = (k+1)
            left   = (k-1)
            left2  = (k-2)
            left3  = (k-3)
            left4  = (k-4)
            if k == 0:
                left   = (Nk_path-1)
                left2  = (Nk_path-2)
                left3  = (Nk_path-3)
                left4  = (Nk_path-4)
            elif k == 1 and dk_order >= 4:
                left2  = (Nk_path-1)
                left3  = (Nk_path-2)
                left4  = (Nk_path-3)
            elif k == 2 and dk_order >= 6:
                left3  = (Nk_path-1)
                left4  = (Nk_path-2)
            elif k == 3 and dk_order >= 8:
                left4  = (Nk_path-1)
            elif k == Nk_path-1:
                right4 = 3
                right3 = 2
                right2 = 1
                right  = 0
            elif k == Nk_path-2 and dk_order >= 4:
                right4 = 2
                right3 = 1
                right2 = 0
            elif k == Nk_path-3 and dk_order >= 6:
                right4 = 1
                right3 = 0
            elif k == Nk_path-4 and dk_order >= 8:
                right4 = 0

            wr = dipole_in_path[k, :, :]*electric_f

            for i in range(n):
                for j in range(n):
                    if i != j:
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
                        if i == j and nbar != i:
                            x[k*(n**2) + i*n + j] += 2 * np.imag( y[k*(n**2) + i*n + nbar] * wr[nbar, i] )
                        else:
                            x[k*(n**2) + i*n + j] += -1j * ( y[k*(n**2) + i*n + nbar] * wr[nbar, j] - wr[i, nbar] * y[k*(n**2) + nbar*n + j])

        x[-1] = -electric_f

        return x

    @conditional_njit(type_complex_np)
    def __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

        pathlen = hpex[:, 0, 0].size

        xderivative = np.empty((pathlen, n, n), dtype=type_complex_np)
        yderivative = np.empty((pathlen, n, n), dtype=type_complex_np)

        eplusx, wfplusx = diagonalize_path(hpex)
        eminusx, wfminusx = diagonalize_path(hmex)
        eplusy, wfplusy = diagonalize_path(hpey)
        eminusy, wfminusy = diagonalize_path(hmey)

        eplus2x, wfplus2x = diagonalize_path(hp2ex)
        eminus2x, wfminus2x = diagonalize_path(hm2ex)
        eplus2y, wfplus2y = diagonalize_path(hp2ey)
        eminus2y, wfminus2y = diagonalize_path(hm2ey)

        eplus3x, wfplus3x = diagonalize_path(hp3ex)
        eminus3x, wfminus3x = diagonalize_path(hm3ex)
        eplus3y, wfplus3y = diagonalize_path(hp3ey)
        eminus3y, wfminus3y = diagonalize_path(hm3ey)

        eplus4x, wfplus4x = diagonalize_path(hp4ex)
        eminus4x, wfminus4x = diagonalize_path(hm4ex)
        eplus4y, wfplus4y = diagonalize_path(hp4ey)
        eminus4y, wfminus4y = diagonalize_path(hm4ey)

        xderivative = (1/280*(wfminus4x - wfplus4x) + 4/105*( wfplus3x - wfminus3x ) + 1/5*( wfminus2x - wfplus2x ) + 4/5*( wfplusx - wfminusx) )/epsilon
        yderivative = (1/280*(wfminus4y - wfplus4y) + 4/105*( wfplus3y - wfminus3y ) + 1/5*( wfminus2y - wfplus2y ) + 4/5*( wfplusy - wfminusy ) )/epsilon
        ederivx = (1/280*(eminus4x - eplus4x) + 4/105*( eplus3x - eminus3x ) + 1/5*( eminus2x - eplus2x ) + 4/5*( eplusx - eminusx) )/epsilon
        ederivy = (1/280*(eminus4y - eplus4y) + 4/105*( eplus3y - eminus3y ) + 1/5*( eminus2y - eplus2y ) + 4/5*( eplusy - eminusy ) )/epsilon

        return xderivative, yderivative, ederivx, ederivy

    @conditional_njit(type_complex_np)
    def diagonalize_path(h_in_path):

        pathlen = h_in_path[:, 0, 0].size

        e_path = np.empty((pathlen, n), dtype=type_real_np)
        wf_path = np.empty((pathlen, n, n), dtype=type_complex_np)

        for i in range(pathlen):
            e_path[i], wf_buff = lin.eigh(h_in_path[i, :, :])
            if degenerate_eigenvalues:
                for j in range(int(n/2)):
                    wf1 = np.copy(wf_buff[:, 2*j])
                    wf2 = np.copy(wf_buff[:, 2*j+1])
                    wf_buff[:, 2*j] *= wf2[n-2]
                    wf_buff[:, 2*j] -= wf1[n-2]*wf2
                    wf_buff[:, 2*j+1] *= wf1[n-1]
                    wf_buff[:, 2*j+1] -= wf2[n-1]*wf1
            wf_gauged_entry = np.copy(wf_buff[gidx, :])
            wf_buff[gidx, :] = np.abs(wf_gauged_entry)
            wf_buff[~(np.arange(n) == gidx)] *= np.exp(1j*np.angle(wf_gauged_entry.conj()))
            wf_path[i] = wf_buff

        return e_path, wf_path

    @conditional_njit(type_complex_np)
    def dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

        pathlen = h_in_path[:, 0, 0].size

        dx_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
        dy_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
        
        if not dm_dynamics_method == 'semiclassics':
            _buf, wf_path = diagonalize_path(h_in_path)
            dwfkx_path, dwfky_path, _buf, _buf = __derivative_path(hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
                                                                        hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

            for i in range(pathlen):
                dx_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfkx_path[i, :, :])
                dy_path[i, :, :] = -1j*np.conjugate(wf_path[i, :, :]).T.dot(dwfky_path[i, :, :])

        return dx_path, dy_path

    @conditional_njit(type_complex_np)
    def pre_velocity_nband(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path

        e_in_path, wf_in_path = diagonalize_path(h_in_path)
        dipole_path_x, dipole_path_y = dipole_path(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, \
                                                        hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

        dipole_in_path = E_dir[0]*dipole_path_x + E_dir[1]*dipole_path_y
        return e_in_path, dipole_in_path

    @conditional_njit(type_complex_np)
    def make_x(t, y, kpath, dipole_in_path, e_in_path, y0, dk, h_in_path, \
                hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey):

        e_in_path, dipole_in_path = pre_velocity_nband(h_in_path, hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, \
                                                        hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

        # x != y(t+dt)
        x = np.zeros(np.shape(y), dtype=type_complex_np)

        electric_f = electric_field(t)

        Nk_path = kpath.shape[0]
        for k in range(Nk_path):

            wr = dipole_in_path[k, :, :]*electric_f

            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[k*(n**2) + i*n + j] += -1j * ( e_in_path[k, i] - e_in_path[k, j] ) * y[k*(n**2) + i*n + j]

                    if i == j:
                        x[k*(n**2) + i*n + j] += - gamma1 * (y[k*(n**2) + i*n + j] - y0[k*(n**2) + i*n + j])
                    else:
                        x[k*(n**2) + i*n + j] += - gamma2 * y[k*(n**2) + i*n + j]
                    for nbar in range(n):
                        if i == j and nbar != i:
                            x[k*(n**2) + i*n + j] += 2 * np.imag( y[k*(n**2) + i*n + nbar] * wr[nbar, i] )
                        else:
                            x[k*(n**2) + i*n + j] += -1j * ( y[k*(n**2) + i*n + nbar] * wr[nbar, j] - wr[i, nbar] * y[k*(n**2) + nbar*n + j])

        x[-1] = -electric_f

        return x

    #@conditional_njit(type_complex_np)
    def fvelocity(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        """
            function that multiplies the block-structure of the matrices of the RHS
            of the SBE with the solution vector
        """
        # evaluate hfjit before using numba
        path_after_shift = np.copy(kpath)

        path_after_shift[:, 0] = kpath[:, 0] + E_dir[0]* y[-1].real
        path_after_shift[:, 1] = kpath[:, 1] + E_dir[1]* y[-1].real

        pathlen = kpath[:, 0].size
        h_in_path = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hpex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hmex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hpey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hmey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm2ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm2ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm3ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm3ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm4ex = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hp4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)
        hm4ey = np.zeros((pathlen, n, n), dtype=type_complex_np)

        for i in range(n):
            for j in range(n):
                for k in range(pathlen):
                    kx = path_after_shift[k, 0]
                    ky = path_after_shift[k, 1]
                    h_in_path[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky)
                    hpex[k, i, j] = sys.hfjit[i][j](kx=kx-P.epsilon, ky=ky)
                    hmex[k, i, j] = sys.hfjit[i][j](kx=kx+P.epsilon, ky=ky)
                    hpey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-P.epsilon)
                    hmey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+P.epsilon)
                    hp2ex[k, i, j] = sys.hfjit[i][j](kx=kx-2*P.epsilon, ky=ky)
                    hm2ex[k, i, j] = sys.hfjit[i][j](kx=kx+2*P.epsilon, ky=ky)
                    hp2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-2*P.epsilon)
                    hm2ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+2*P.epsilon)
                    hp3ex[k, i, j] = sys.hfjit[i][j](kx=kx-3*P.epsilon, ky=ky)
                    hm3ex[k, i, j] = sys.hfjit[i][j](kx=kx+3*P.epsilon, ky=ky)
                    hp3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-3*P.epsilon)
                    hm3ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+3*P.epsilon)
                    hp4ex[k, i, j] = sys.hfjit[i][j](kx=kx-4*P.epsilon, ky=ky)
                    hm4ex[k, i, j] = sys.hfjit[i][j](kx=kx+4*P.epsilon, ky=ky)
                    hp4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky-4*P.epsilon)
                    hm4ey[k, i, j] = sys.hfjit[i][j](kx=kx, ky=ky+4*P.epsilon)

        x = make_x(t, y, kpath, dipole_in_path, e_in_path, y0, dk, h_in_path, \
            hpex, hmex, hp2ex, hm2ex, hp3ex, hm3ex, hp4ex, hm4ex, hpey, hmey, hp2ey, hm2ey, hp3ey, hm3ey, hp4ey, hm4ey)

        return x

    freturn = None
    if system == 'bandstructure' and gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity_custom_bs
    elif gauge == 'length':
        print("Using length gauge")
        freturn = flength
    elif gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")

    def f(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        return freturn(t, y, kpath, dipole_in_path, e_in_path, y0, dk)

    return f
