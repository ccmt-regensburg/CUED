from numba import njit
import numpy as np
from hfsbe.utility import evaluate_njit_matrix as evmat


def make_velocity_zeeman_solver(sys, dipole_k, dipole_B, gamma1, gamma2, E_dir,
                                electric_field, zeeman_field):
    # Wire the energies
    evf = sys.efjit[0]
    ecf = sys.efjit[1]

    # Wire the dipoles
    # kx-parameter
    di_00xf = dipole_k.Axfjit[0][0]
    di_01xf = dipole_k.Axfjit[0][1]
    di_11xf = dipole_k.Axfjit[1][1]

    # ky-parameter
    di_00yf = dipole_k.Ayfjit[0][0]
    di_01yf = dipole_k.Ayfjit[0][1]
    di_11yf = dipole_k.Ayfjit[1][1]

    # Mx - Zeeman z parameter
    di_00mxf = dipole_B.Mxfjit[0][0]
    di_01mxf = dipole_B.Mxfjit[0][1]
    di_11mxf = dipole_B.Mxfjit[1][1]

    di_00myf = dipole_B.Myfjit[0][0]
    di_01myf = dipole_B.Myfjit[0][1]
    di_11myf = dipole_B.Myfjit[1][1]

    di_00mzf = dipole_B.Mzfjit[0][0]
    di_01mzf = dipole_B.Mzfjit[0][1]
    di_11mzf = dipole_B.Mzfjit[1][1]

    @njit
    def fvelocity(t, y, kpath, dk, y0):

        # Preparing system parameters, energies, dipoles

        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]
        k_shift = y[-1].real

        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ev = evf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ec = ecf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        ecv_in_path = ec - ev

        di_00x = di_00xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01x = di_01xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11x = di_11xf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_00y = di_00yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_01y = di_01yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
        di_11y = di_11yf(kx=kx, ky=ky, m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

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
            x[i] = 2*(wr*y[i+1]).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr_c*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(wr*y[i+1]).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f
        return x

    def f(t, y, kpath, dk, y0):
        return fvelocity(t, y, kpath, dk, y0)

    return f


def emission_exact_zeeman(sys, paths, tarr, solution, E_dir, A_field,
                          zeeman_field):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # n_time_steps = np.size(solution[0, 0, :, 0])
    n_time_steps = np.size(tarr)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time, t in enumerate(tarr):
        m_zee = zeeman_field(t)
        mx = m_zee[0]
        my = m_zee[1]
        mz = m_zee[2]
        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]

            kx_in_path_shifted = kx_in_path - A_field[i_time]*E_dir[0]
            ky_in_path_shifted = ky_in_path - A_field[i_time]*E_dir[1]

            h_deriv_x = evmat(sys.hderivfjit[0],
                              kx=kx_in_path_shifted, ky=ky_in_path_shifted,
                              m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
            h_deriv_y = evmat(sys.hderivfjit[1],
                              kx=kx_in_path_shifted, ky=ky_in_path_shifted,
                              m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.Uf(kx=kx_in_path, ky=ky_in_path,
                       m_zee_x=mx, m_zee_y=my, m_zee_z=mz)
            U_h = sys.Uf_h(kx=kx_in_path, ky=ky_in_path,
                           m_zee_x=mx, m_zee_y=my, m_zee_z=mz)

            for i_k in range(np.size(kx_in_path)):

                dH_U_E_dir = np.matmul(h_deriv_E_dir[:, :, i_k], U[:, :, i_k])
                U_h_H_U_E_dir = np.matmul(U_h[:, :, i_k], dH_U_E_dir)

                dH_U_ortho = np.matmul(h_deriv_ortho[:, :, i_k], U[:, :, i_k])
                U_h_H_U_ortho = np.matmul(U_h[:, :, i_k], dH_U_ortho)

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho

