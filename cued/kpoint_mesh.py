import numpy as np
###############################################################################
# K-Point meshes
###############################################################################
def rect_mesh(P):
    '''
    Create a rectangular mesh
    '''
    # Number of kpoints in E-field direction and orthogonal to the E-field
    E_dir = P.E_dir
    Nk_E_dir_integer = P.Nk1
    Nk_E_dir         = P.type_real_np(Nk_E_dir_integer)
    Nk_ortho_integer = P.Nk2
    Nk_ortho         = P.type_real_np(Nk_ortho_integer)

    # length of the rectangle in E-field direction and orthogonal to it
    length_BZ_E_dir = P.type_real_np(P.length_BZ_E_dir)
    length_BZ_ortho = P.type_real_np(P.length_BZ_ortho)

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_E_dir)),
                               0.5 - (1/(2*Nk_E_dir)), num=Nk_E_dir_integer, dtype=P.type_real_np)
    beta_array  = np.linspace(-0.5 + (1/(2*Nk_ortho)),
                               0.5 - (1/(2*Nk_ortho)), num=Nk_ortho_integer, dtype=P.type_real_np)

    vec_k_E_dir = length_BZ_E_dir*E_dir
    vec_k_ortho = length_BZ_ortho*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
    for beta in beta_array:

        # Container for a single path
        path = []
        for alpha in alpha_array:
            # Create a k-point
            kpoint = alpha*vec_k_E_dir + beta*vec_k_ortho

            mesh.append(kpoint)
            path.append(kpoint)

        # Append the a1'th path to the paths array
        paths.append(path)

    dk = length_BZ_E_dir/Nk_E_dir
    if P.Nk2 == 1:
        kweight = length_BZ_E_dir/Nk_E_dir * two_pi_factor(P)
    else:
        kweight = length_BZ_E_dir/Nk_E_dir * length_BZ_ortho/Nk_ortho * two_pi_factor(P)

    return dk, kweight, np.array(paths), np.array(mesh)

def hex_mesh(P):
    '''
    Create a hexagonal mesh
    '''

    b1 = (2*np.pi/(P.a*np.sqrt(3)))*np.array([np.sqrt(3), -1], dtype=P.type_real_np)
    b2 = (4*np.pi/(P.a*np.sqrt(3)))*np.array([0, 1], dtype=P.type_real_np)

    def is_in_hex(p):
        # Returns true if the point is in the hexagonal BZ.
        # Checks if the absolute values of x and y components of p are within
        # the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*P.a)) and
                (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*P.a)))

    def reflect_point(p):
        x = p[0]
        y = p[1]
        if y > 2*np.pi/(np.sqrt(3)*P.a):
            # Crosses top
            p -= b2
        elif y < -2*np.pi/(np.sqrt(3)*P.a):
            # Crosses bottom
            p += b2
        elif np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*P.a):
            # Crosses top-right
            p -= b1 + b2
        elif -np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*P.a):
            # Crosses bot-right
            p -= b1
        elif np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*P.a):
            # Crosses bot-left
            p += b1 + b2
        elif -np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*P.a):
            # Crosses top-left
            p += b1
        return p

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []
    angle = 0

    if P.angle_inc_E_field != None:
        if P.angle_inc_E_field <= 15:
            P.align = 'K'
            angle = np.radians(0)
        elif P.angle_inc_E_field <= 45:
            P.align = 'M'
            angle = np.radians(60)
        elif P.angle_inc_E_field <= 75:
            P.align = 'K'
            angle = np.radians(60)
        elif P.angle_inc_E_field <= 105:
            P.align = 'M'
            angle = np.radians(120)
        elif P.angle_inc_E_field <= 135:
            P.align = 'K'
            angle = np.radians(120)
        elif P.angle_inc_E_field <= 165:
            P.align = 'M'
            angle = np.radians(180)
        elif P.angle_inc_E_field <= 195:
            P.align = 'K'
            angle = np.radians(180)
        elif P.angle_inc_E_field <= 225:
            P.align = 'M'
            angle = np.radians(240)
        elif P.angle_inc_E_field <= 255:
            P.align = 'K'
            angle = np.radians(240)
        elif P.angle_inc_E_field <= 285:
            P.align = 'M'
            angle = np.radians(300)
        elif P.angle_inc_E_field <= 315:
            P.align = 'K'
            angle = np.radians(300)
        elif P.angle_inc_E_field <= 345:
            P.align = 'M'
            angle = np.radians(0)
        else:
            P.algin = 'K'
            angle = np.radians(0)

    # Create the Monkhorst-Pack mesh
    if P.align == 'M':
        if P.Nk2%3 != 0:
            raise RuntimeError("Nk2: " + "{:d}".format(P.Nk2) +
                               " needs to be divisible by 3")
        b_a1 = b1
        b_a2 = (2*np.pi/(3*P.a))*np.array([1, np.sqrt(3)], dtype=P.type_real_np)
        alpha1 = np.linspace(-0.5 + (1/(2*P.Nk1)), 0.5 - (1/(2*P.Nk1)), num=P.Nk1, dtype=P.type_real_np)
        alpha2 = np.linspace(-1.0 + (1.5/(2*P.Nk2)), 0.5 - (1.5/(2*P.Nk2)), num=P.Nk2, dtype=P.type_real_np)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        b_a1 = np.dot(rotation_matrix, b_a1)
        b_a2 = np.dot(rotation_matrix, b_a2)
        for a2 in alpha2:
            # Container for a single gamma-M path
            path_M = []
            for a1 in alpha1:
                # Create a k-point
                kpoint = a1*b_a1 + a2*b_a2
                # If current point is in BZ, append it to the mesh and path_M
                if is_in_hex(kpoint):
                    mesh.append(kpoint)
                    path_M.append(kpoint)
                # If current point is NOT in BZ, reflect it along
                # the appropriate axis to get it in the BZ, then append.
                else:
                    while not is_in_hex(kpoint):
                        kpoint = reflect_point(kpoint)
                    mesh.append(kpoint)
                    path_M.append(kpoint)
            # Append the a1'th path to the paths array
            paths.append(path_M)
        dk = np.linalg.norm(b_a1)/P.Nk1

    elif P.align == 'K':
        if P.Nk1%3 != 0 or P.Nk1%2 != 0:
            raise RuntimeError("Nk1: " + "{:d}".format(P.Nk1) +
                               " needs to be divisible by 3 and even")
        if P.Nk2%3 != 0 or P.Nk2%2 != 0:
            raise RuntimeError("Nk2: " + "{:d}".format(P.Nk2) +
                               " needs to be divisible by 3 and even")
        b_a1 = 8*np.pi/(P.a*3)*np.array([1, 0], dtype=P.type_real_np)
        b_a2 = 4*np.pi/(P.a*3)*np.array([0, np.sqrt(3)], dtype=P.type_real_np)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        b_a1 = np.dot(rotation_matrix, b_a1)
        b_a2 = np.dot(rotation_matrix, b_a2)
        # Extend over half of the b2 direction and 1.5x the b1 direction
        # (extending into the 2nd BZ to get correct boundary conditions)
        if angle == 0:
            alpha1 = np.linspace(-0.5 + (1.5/(2*P.Nk1)), 1.0 - (1.5/(2*P.Nk1)), P.Nk1, dtype=P.type_real_np)
            alpha2 = np.linspace(0 + (0.5/(2*P.Nk2)), 0.5 - (0.5/(2*P.Nk2)), P.Nk2, dtype=P.type_real_np)
        else:
            alpha1 = np.linspace(0 + (0.75/(2*P.Nk1)), 0.75 - (0.75/(2*P.Nk1)), P.Nk1, dtype=P.type_real_np)
            alpha2 = np.linspace(0 + (1/(2*P.Nk2)), 1 - (1/(2*P.Nk2)), P.Nk2, dtype=P.type_real_np)
        for a2 in alpha2:
            path_K = []
            for a1 in alpha1:
                kpoint = a1*b_a1 + a2*b_a2
                if is_in_hex(kpoint):
                   mesh.append(kpoint)
                   path_K.append(kpoint)
                else:
                   while not is_in_hex(kpoint):
                       kpoint = reflect_point(kpoint)
                   # kpoint -= (2*np.pi/a) * np.array([1, 1/np.sqrt(3)])
                   mesh.append(kpoint)
                   path_K.append(kpoint)
            paths.append(path_K)
        dk = np.linalg.norm(1.5*b_a1)/P.Nk1

    return dk, (3*np.sqrt(3)/2)*(4*np.pi/(P.a*3))**2/P.Nk*two_pi_factor(P), np.array(paths), np.array(mesh)


def two_pi_factor(P):
    if P.num_dimensions == '1':
        exponent = -1
    elif P.num_dimensions == '2':
        exponent = -2
    elif P.num_dimensions == 'automatic':
        if P.Nk2 == 1:
            exponent = -1
        else:
            exponent = -2

    return P.type_real_np((2.0*np.pi)**exponent)
