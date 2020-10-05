import numpy as np
from numba import njit


def make_polarization_path(dipole, pathlen, n_time_steps, E_dir, A_field, gauge):
    dijit_01x = dipole.Axfjit[0][1]
    dijit_01y = dipole.Ayfjit[0][1]

    E_ort = np.array([E_dir[1], -E_dir[0]])

    @njit
    def polarization_path(path, pcv, P_E_dir, P_ortho):
        ##################################################
        # Dipole container
        ##################################################
        d_01x = np.zeros(pathlen, dtype=np.complex128)
        d_01y = np.zeros(pathlen, dtype=np.complex128)

        d_E_dir = np.zeros(pathlen, dtype=np.complex128)
        d_ortho = np.zeros(pathlen, dtype=np.complex128)

        for i_time in range(n_time_steps):
            if gauge == 'length':
                kx_shift = 0
                ky_shift = 0
            if gauge == 'velocity':
                kx_shift = A_field[i_time]*E_dir[0]
                ky_shift = A_field[i_time]*E_dir[1]

            kx_in_path = path[:, 0] + kx_shift
            ky_in_path = path[:, 1] + ky_shift

            d_01x[:] = dijit_01x(kx=kx_in_path, ky=ky_in_path)
            d_01y[:] = dijit_01y(kx=kx_in_path, ky=ky_in_path)

            d_E_dir[:] = d_01x * E_dir[0] + d_01y * E_dir[1]
            d_ortho[:] = d_01x * E_ort[0] + d_01y * E_ort[1]

            P_E_dir[i_time] += 2*np.real(np.sum(d_E_dir * pcv[:, i_time]))
            P_ortho[i_time] += 2*np.real(np.sum(d_ortho * pcv[:, i_time]))

    return polarization_path

def make_current_path(sys, pathlen, n_time_steps, E_dir, A_field, gauge):
    '''
    Calculates the current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)
    '''
    edxjit_v = sys.ederivfjit[0]
    edyjit_v = sys.ederivfjit[1]
    edxjit_c = sys.ederivfjit[2]
    edyjit_c = sys.ederivfjit[3]

    E_ort = np.array([E_dir[1], -E_dir[0]])

    @njit
    def current_path(path, fv, fc, J_E_dir, J_ortho):
        ##################################################
        # E derivative container
        ##################################################
        edx_v = np.zeros(pathlen, dtype=np.float64)
        edy_v = np.zeros(pathlen, dtype=np.float64)
        edx_c = np.zeros(pathlen, dtype=np.float64)
        edy_c = np.zeros(pathlen, dtype=np.float64)

        e_deriv_E_dir_v = np.zeros(pathlen, dtype=np.float64)
        e_deriv_ortho_v = np.zeros(pathlen, dtype=np.float64)
        e_deriv_E_dir_c = np.zeros(pathlen, dtype=np.float64)
        e_deriv_ortho_c = np.zeros(pathlen, dtype=np.float64)

        for i_time in range(n_time_steps):
            if gauge == 'length':
                kx_shift = 0
                ky_shift = 0
                fv_subs = 0
            if gauge == 'velocity':
                kx_shift = A_field[i_time]*E_dir[0]
                ky_shift = A_field[i_time]*E_dir[1]
                fv_subs = 1

            kx_in_path = path[:, 0] + kx_shift
            ky_in_path = path[:, 1] + ky_shift

            edx_v[:] = edxjit_v(kx=kx_in_path, ky=ky_in_path)
            edy_v[:] = edyjit_v(kx=kx_in_path, ky=ky_in_path)
            edx_c[:] = edxjit_c(kx=kx_in_path, ky=ky_in_path)
            edy_c[:] = edyjit_c(kx=kx_in_path, ky=ky_in_path)


            e_deriv_E_dir_v[:] = edx_v * E_dir[0] + edy_v * E_dir[1]
            e_deriv_ortho_v[:] = edx_v * E_ort[0] + edy_v * E_ort[1]
            e_deriv_E_dir_c[:] = edx_c * E_dir[0] + edy_c * E_dir[1]
            e_deriv_ortho_c[:] = edx_c * E_ort[0] + edy_c * E_ort[1]

            J_E_dir[i_time] += np.sum(e_deriv_E_dir_v * (fv[:, i_time].real - fv_subs)) + \
                np.sum(e_deriv_E_dir_c * fc[:, i_time].real)
            J_ortho[i_time] += np.sum(e_deriv_ortho_v * (fv.real[:, i_time] - fv_subs)) + \
                np.sum(e_deriv_ortho_c * fc[:, i_time].real)

    return current_path

def make_emission_exact_path(sys, pathlen, n_time_steps, E_dir, A_field, gauge, do_semicl, curvature, E_field):
    """
    Construct a function that calculates the emission for the system solution per path

    Parameters
    ----------
    sys : TwoBandSystem
        Hamiltonian and related functions
    pathlen : int
        Length of one path
    n_time_steps : int
        Number of time steps
    E_dir : list
        Direction of the electric field
    A_field : np.ndarray
        Vector potential integrated from electric field
    gauge : string
        System gauge 'length' or 'velocity'
    do_semicl : bool
        if semiclassical calculation should be done
    curvature : SymbolicCurvature
        Curvature is only needed for semiclassical calculation
    electric_f : np.ndarray
        Electric field is only needed for semiclassical calculation
    """

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

    if do_semicl:
        Bcurv_00 = curvature.Bfjit[0][0]
        Bcurv_11 = curvature.Bfjit[1][1]

    E_ort = np.array([E_dir[1], -E_dir[0]])

    @njit
    def emission_exact_path(path, solution, I_E_dir, I_ortho):
        ##########################################################
        # H derivative container
        ##########################################################
        h_deriv_x = np.empty((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_y = np.empty((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_E_dir = np.empty((pathlen, 2, 2), dtype=np.complex128)
        h_deriv_ortho = np.empty((pathlen, 2, 2), dtype=np.complex128)

        ##########################################################
        # Wave function container
        ##########################################################
        U = np.empty((pathlen, 2, 2), dtype=np.complex128)
        U_h = np.empty((pathlen, 2, 2), dtype=np.complex128)

        ##########################################################
        # Berry curvature container
        ##########################################################
        if do_semicl:
            Bcurv = np.empty((pathlen, 2), dtype=np.complex128)

        # I_E_dir is of size (number of time steps)
        for i_time in range(n_time_steps):
            if gauge == 'length':
                kx_shift = 0
                ky_shift = 0
                fv_subs = 0
            if gauge == 'velocity':
                kx_shift = A_field[i_time]*E_dir[0]
                ky_shift = A_field[i_time]*E_dir[1]
                fv_subs = 1

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

            if do_semicl:
                Bcurv[:, 0] = Bcurv_00(kx=kx_in_path, ky=ky_in_path)
                Bcurv[:, 1] = Bcurv_11(kx=kx_in_path, ky=ky_in_path)

            for i_k in range(pathlen):

                dH_U_E_dir = h_deriv_E_dir[i_k] @ U[i_k]
                U_h_H_U_E_dir = U_h[i_k] @ dH_U_E_dir

                dH_U_ortho = h_deriv_ortho[i_k] @ U[i_k]
                U_h_H_U_ortho = U_h[i_k] @ dH_U_ortho

                I_E_dir[i_time] += U_h_H_U_E_dir[0, 0].real\
                    * (solution[i_k, i_time, 0].real - fv_subs)
                I_E_dir[i_time] += U_h_H_U_E_dir[1, 1].real\
                    * solution[i_k, i_time, 3].real
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                             * solution[i_k, i_time, 2])

                I_ortho[i_time] += U_h_H_U_ortho[0, 0].real\
                    * (solution[i_k, i_time, 0].real - fv_subs)
                I_ortho[i_time] += U_h_H_U_ortho[1, 1].real\
                    * solution[i_k, i_time, 3].real
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                             * solution[i_k, i_time, 2])

                if do_semicl:
                    I_ortho[i_time] += E_field[i_time] * (Bcurv[i_k, 0].real + Bcurv[i_k, 1].real)

    return emission_exact_path
