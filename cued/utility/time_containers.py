import numpy as np
from cued.fields import make_electric_field

def time_containers(P, electric_field_function):
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

    # Initialize electric_field, create rhs of ode and initialize solver

    if electric_field_function is None:
        T.electric_field = make_electric_field(P)
    else:
        T.electric_field = electric_field_function

    return T