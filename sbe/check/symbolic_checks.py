"""
Functional checks of different properties that the symbolic
expressions like Hamiltonians, energies, wave functions, dipole
moments etc. should have
"""
import sympy as sp


def eigensystem(h, e, wf):
    """
    Check the input eigensystem for correctness regarding energies and
    orthonormality
    """
    if (len(wf) != 2):
        raise RuntimeError("Error: You are missing either U or U^dagger in "
                           "the wave function list.")

    if (len(e) != wf[0].shape[0]):
        raise RuntimeError("Error: You do not have as many wave functions as "
                           "energies.")

    U = wf[0]
    U_h = wf[1]
    dim = wf[0].shape[0]

    # Check orthonormality between all combinations
    # orthonormal_check is unit matrix if everything is alright
    orthonormal_check = sp.simplify(U*U_h)

    if (orthonormal_check == sp.eye(dim)):
        print("Wave functions are orthonormal.")
    else:
        raise RuntimeError("The wave functions are not orthonormal, this is "
                           " the result of U*U_h: ", orthonormal_check)

    # TODO Add check to determine if it is a eigenstate
