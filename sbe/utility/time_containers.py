import numpy as np

def time_containers(P):
    class Time():
        pass

    T  = Time()

    T.t = np.zeros(P.Nt, dtype=P.type_real_np)
    
    T.solution = np.zeros((P.Nk1, P.n, P.n), dtype=P.type_real_np)
    T.solution_y_vec = np.zeros((((P.n)**2)*(P.Nk1)+1), dtype=P.type_complex_np)
    
    if P.save_full:
        solution_full = np.empty((P.Nk1, P.Nk2, P.Nt, P.n, P.n), dtype=P.type_complex_np)
        

    T.A_field = np.zeros(P.Nt, dtype=P.type_real_np)
    T.E_field = np.zeros(P.Nt, dtype=P.type_real_np)

    T.J_exact_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
    T.J_exact_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

    if P.save_approx:
        T.J_intra_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        T.J_intra_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

        T.P_inter_E_dir = np.zeros(P.Nt, dtype=P.type_real_np)
        T.P_inter_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

        T.J_anom_ortho = np.zeros(P.Nt, dtype=P.type_real_np)

    #if P.zeeman:
    #    T.Zee_field = np.zeros((P.Nt, 3), dtype=P.type_real_np)

    return T