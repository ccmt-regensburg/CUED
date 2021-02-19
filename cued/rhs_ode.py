import numpy as np
from cued.utility import conditional_njit


def make_rhs_ode_2_band(sys, dipole, E_dir, electric_field, P):
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
    do_semicl = P.do_semicl
    gauge = P.gauge

    if P.hamiltonian_evaluation == 'ana':
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

    @conditional_njit(P.type_complex_np)
    def flength(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
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
                
            x[i+2] = x[i+1].conjugate()

        x[-1] = -electric_f
        return x

    @conditional_njit(P.type_complex_np)
    def pre_velocity(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if do_semicl:
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
    def fvelocity(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
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
    def f(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        return freturn(t, y, kpath, dipole_in_path, e_in_path, y0, dk)

    return f

def make_rhs_ode_n_band(E_dir, electric_field, P):
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
    dk_order = P.dk_order
    gauge = P.gauge
    n = P.n

    @conditional_njit(P.type_complex_np)
    def fnumba(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
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
                        if i == j:
                            x[k*(n**2) + i*n + j] += 2 * np.imag( y[k*(n**2) + i*n + nbar] * wr[nbar, i] )
                        else:
                            x[k*(n**2) + i*n + j] += -1j * ( y[k*(n**2) + i*n + nbar] * wr[nbar, j] - wr[i, nbar] * y[k*(n**2) + nbar*n + j])

        x[-1] = -electric_f

        return x
    
    freturn = None
    if gauge == 'length':
        print("Using length gauge")
        freturn = fnumba
    elif gauge == 'velocity':
        print("Using velocity gauge")
        raise AttributeError("Velocity gauge is not available for n-band solver")
    else:
        raise AttributeError("You have to either assign velocity or length gauge")

    def f(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        return fnumba(t, y, kpath, dipole_in_path, e_in_path, y0, dk)

    return f
