import sympy as sp
import numpy as np

def eigensystem(h, e, wf):
    """
    Check the input eigensystem for correctness regarding energies and
    orthonormality
    """

    if (e.shape[0] != wf.shape[0] or wf.shape[1] != 2):
        raise RuntimeError("Error: You do not have as many wave functions as "
                           "energies or the complex conjugates are missing.")

    # Check orthonormality between all combinations
    orthonormal_check = wf[0, :].dot(wf[1, :])
    vsimplify = np.vectorize(sp.simplify)
    orthonormal_check = vsimplify(orthonormal_check)

    # orthonormal_check is unit matrix if everything is alright
    orthonormal_bool = np.equal(orthonormal_check, np.eye(e.shape[0]))
    
    if (np.all(orthonormal_bool)):
        print("Wave functions are orthonormal.")
    else:
        print("The following wave functions are not normalised or orthogonal "
              " to each other:")
        for pair in np.where(orthonormal_bool == False):
            print("Index %d to %d", %(pair[0], pair[1]))
        

    #TODO Add check to determine if it is a eigenstate
