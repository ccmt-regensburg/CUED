import numpy as np
from numba import njit

def polarization(dip, paths, pcv, E_dir):
    '''
    Calculates the polarization as: P(t) = sum_n sum_m sum_k [d_nm(k)p_nm(k)]
    Dipole term currently a crude model to get a vector polarization
    '''
    E_ort = np.array([E_dir[1], -E_dir[0]])

    d_E_dir, d_ortho = [], []
    for path in paths:

        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Evaluate the dipole moments in path
        di_01x = dip.Axfjit[0][1](kx=kx_in_path, ky=ky_in_path)
        di_01y = dip.Ayfjit[0][1](kx=kx_in_path, ky=ky_in_path)

        # Append the dot product d.E
        d_E_dir.append(di_01x*E_dir[0] + di_01y*E_dir[1])
        d_ortho.append(di_01x*E_ort[0] + di_01y*E_ort[1])

    d_E_dir_swapped = np.swapaxes(d_E_dir, 0, 1)
    d_ortho_swapped = np.swapaxes(d_ortho, 0, 1)

    P_E_dir = 2*np.real(np.tensordot(d_E_dir_swapped, pcv, 2))
    P_ortho = 2*np.real(np.tensordot(d_ortho_swapped, pcv, 2))

    return P_E_dir, P_ortho


def current(sys, paths, fv, fc, t, alpha, E_dir):
    '''
    Calculates the current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)
    '''
    E_ort = np.array([E_dir[1], -E_dir[0]])

    # Calculate the gradient analytically at each k-point
    J_E_dir, J_ortho = [], []
    jc_E_dir, jc_ortho, jv_E_dir, jv_ortho = [], [], [], []
    for path in paths:
        path = np.array(path)
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        evdx = sys.ederivfjit[0](kx=kx_in_path, ky=ky_in_path)
        evdy = sys.ederivfjit[1](kx=kx_in_path, ky=ky_in_path)
        ecdx = sys.ederivfjit[2](kx=kx_in_path, ky=ky_in_path)
        ecdy = sys.ederivfjit[3](kx=kx_in_path, ky=ky_in_path)

        # 0: v, x   1: v,y   2: c, x  3: c, y
        jc_E_dir.append(ecdx*E_dir[0] + ecdy*E_dir[1])
        jc_ortho.append(ecdx*E_ort[0] + ecdy*E_ort[1])
        jv_E_dir.append(evdx*E_dir[0] + evdy*E_dir[1])
        jv_ortho.append(evdx*E_ort[0] + evdy*E_ort[1])

    jc_E_dir = np.swapaxes(jc_E_dir, 0, 1)
    jc_ortho = np.swapaxes(jc_ortho, 0, 1)
    jv_E_dir = np.swapaxes(jv_E_dir, 0, 1)
    jv_ortho = np.swapaxes(jv_ortho, 0, 1)

    # tensordot for contracting the first two indices (2 kpoint directions)
    J_E_dir = np.tensordot(jc_E_dir, fc, 2) + np.tensordot(jv_E_dir, fv, 2)
    J_ortho = np.tensordot(jc_ortho, fc, 2) + np.tensordot(jv_ortho, fv, 2)

    # Return the real part of each component
    return np.real(J_E_dir), np.real(J_ortho)


def make_emission_exact(sys, paths, solution, E_dir, A_field, gauge):
    hderivx = sys.hderivfjit[0]
    hdx_00 = hderivx[0][0]
    hdx_01 = hderivx[0][1]
    hdx_10 = hderivx[1][0]
    hdx_11 = hderivx[1][1]

    hderivy = sys.hderivfjit[1]
    hdy_00 = hderivy[0][0]
    hdy_01 = hderivy[0][1]
    hdy_10 = hderivy[1][0]
    hdy_11 = hderivy[1][1]

    Ujit = sys.Ujit
    U_00 = Ujit[0][0]
    U_01 = Ujit[0][1]
    U_10 = Ujit[1][0]
    U_11 = Ujit[1][1]

    Ujit_h = sys.Ujit_h
    U_h_00 = Ujit_h[0][0]
    U_h_01 = Ujit_h[0][1]
    U_h_10 = Ujit_h[1][0]
    U_h_11 = Ujit_h[1][1]

    n_time_steps = np.size(solution, axis=2)
    pathlen = np.size(solution, axis=0)
    E_ort = np.array([E_dir[1], -E_dir[0]])

    @njit
    def emission_exact():
        ##########################################################
        # H derivative container
        ##########################################################
        h_deriv_x = np.zeros((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_y = np.zeros((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_E_dir = np.zeros((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_ortho = np.zeros((pathlen, 2, 2), dtype=np.complex128)

        ##########################################################
        # Wave function container
        ##########################################################
        U = np.zeros((pathlen, 2, 2), dtype=np.complex128)
        U_h = np.zeros((pathlen, 2, 2), dtype=np.complex128)

        ##########################################################
        # Solution container
        ##########################################################
        I_E_dir = np.zeros(n_time_steps, dtype=np.float64)
        I_ortho = np.zeros(n_time_steps, dtype=np.float64)

        # I_E_dir is of size (number of time steps)
        for i_time in range(n_time_steps):
            if (gauge == 'length'):
                kx_shift = 0
                ky_shift = 0
                fv_subs = 0
            if (gauge == 'velocity'):
                kx_shift = A_field[i_time]*E_dir[0]
                ky_shift = A_field[i_time]*E_dir[1]
                fv_subs = 1

            for i_path, path in enumerate(paths):
                kx_in_path = path[:, 0] + kx_shift
                ky_in_path = path[:, 1] + ky_shift

                h_deriv_x[:, 0, 0] = hdx_00(kx=kx_in_path, ky=ky_in_path)
                h_deriv_x[:, 0, 1] = hdx_01(kx=kx_in_path, ky=ky_in_path)
                h_deriv_x[:, 1, 0] = hdx_10(kx=kx_in_path, ky=ky_in_path)
                h_deriv_x[:, 1, 1] = hdx_11(kx=kx_in_path, ky=ky_in_path)

                h_deriv_y[:, 0, 0] = hdy_00(kx=kx_in_path, ky=ky_in_path)
                h_deriv_y[:, 0, 1] = hdy_01(kx=kx_in_path, ky=ky_in_path)
                h_deriv_y[:, 1, 0] = hdy_10(kx=kx_in_path, ky=ky_in_path)
                h_deriv_y[:, 1, 1] = hdy_11(kx=kx_in_path, ky=ky_in_path)

                h_deriv_E_dir[:, :, :] = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
                h_deriv_ortho[:, :, :] = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

                U[:, 0, 0] = U_00(kx=kx_in_path, ky=ky_in_path)
                U[:, 0, 1] = U_01(kx=kx_in_path, ky=ky_in_path)
                U[:, 1, 0] = U_10(kx=kx_in_path, ky=ky_in_path)
                U[:, 1, 1] = U_11(kx=kx_in_path, ky=ky_in_path)

                U_h[:, 0, 0] = U_h_00(kx=kx_in_path, ky=ky_in_path)
                U_h[:, 0, 1] = U_h_01(kx=kx_in_path, ky=ky_in_path)
                U_h[:, 1, 0] = U_h_10(kx=kx_in_path, ky=ky_in_path)
                U_h[:, 1, 1] = U_h_11(kx=kx_in_path, ky=ky_in_path)

                for i_k in range(pathlen):

                    dH_U_E_dir = h_deriv_E_dir[i_k] @ U[i_k]
                    U_h_H_U_E_dir = U_h[i_k] @ dH_U_E_dir

                    dH_U_ortho = h_deriv_ortho[i_k] @ U[i_k]
                    U_h_H_U_ortho = U_h[i_k] @ dH_U_ortho

                    I_E_dir[i_time] += U_h_H_U_E_dir[0, 0].real\
                        * (solution[i_k, i_path, i_time, 0].real - fv_subs)
                    I_E_dir[i_time] += U_h_H_U_E_dir[1, 1].real\
                        * solution[i_k, i_path, i_time, 3].real
                    I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                                 * solution[i_k, i_path, i_time, 2])

                    I_ortho[i_time] += U_h_H_U_ortho[0, 0].real\
                        * (solution[i_k, i_path, i_time, 0].real - fv_subs)
                    I_ortho[i_time] += U_h_H_U_ortho[1, 1].real\
                        * solution[i_k, i_path, i_time, 3].real
                    I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                                 * solution[i_k, i_path, i_time, 2])

        return I_E_dir, I_ortho

    return emission_exact
