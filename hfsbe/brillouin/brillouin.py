import numpy as np


def to_reciprocal_coordinates(kx, ky, b1, b2):
    kmat = np.vstack((kx, ky))
    bmat = np.array([[b1[0], b2[0]],
                     [b1[1], b2[1]]])
    amat = np.linalg.solve(bmat, kmat)
    return amat[0], amat[1]


def to_cartesian_coordinates(a1, a2, b1, b2):
    amat = np.vstack((a1, a2))
    bmat = np.array([[b1[0], b2[0]],
                     [b1[1], b2[1]]])
    kmat = bmat.dot(amat)
    return kmat[0], kmat[1]


def in_brillouin(kx, ky, b1, b2, eps=10e-10):
    """
    Return a collection of k-points inside a BZ determined
    by the reciprocal lattice vectors b1, b2

    Parameters:
    kx, ky : np.ndarray
        array of all point combinations
    b1, b2 : np.ndarray
        reciprocal lattice vector
    eps : float
        Threshold to include/exclude boundary points

    Returns:
    kx, ky : np.ndarray
        array of all kpoints inside Brillouin zone
    """
    a1, a2 = to_reciprocal_coordinates(kx, ky, b1, b2)

    is_less_a1 = np.abs(a1) <= 0.5 + eps
    is_less_a2 = np.abs(a2) <= 0.5 + eps
    is_inzone = np.logical_and(is_less_a1, is_less_a2)

    return is_inzone


def intersect_brillouin(kx, ky, b1, b2):
    """
    Find the intersection points of a line through a point (kx, ky)
    with the brillouin zone boundary
    """
    a1, a2 = to_reciprocal_coordinates(kx, ky, b1, b2)
    try:
        beta = np.array([0.5/a1, 0.5/a2])
    except RuntimeWarning:
        pass

    beta = np.min(np.abs(beta), axis=0)
    a1_b, a2_b = beta*a1, beta*a2
    return to_cartesian_coordinates(a1_b, a2_b, b1, b2)


def evaluate_dipole(dipole, kx, ky, b1, b2, hamr=None,
                    eps=10e-10, **fkwargs):
    """
    Evaluates a function on a given kgrid defined by kx and ky.
    If hamiltonian_radius is given it will interpolate the function
    linearly from the given radius to the edge of the Brillouin zone
    to the other end of the circle.
    """
    if (not np.all(in_brillouin(kx, ky, b1, b2, eps))):
        raise RuntimeError("Error: Not all the given k-points are inside "
                           "the Brillouin zone.")

    if (hamr is None):
        # No hamiltonian region given -> defined in entire bz
        return dipole(kx=kx, ky=ky, **fkwargs)
    else:
        # Regular evaluation in circle
        # # Find output shape of the function
        testval = dipole(kx=kx[0], ky=ky[0], **fkwargs)
        outshape = np.shape(testval)

        result = np.empty(outshape + (kx.size, ), dtype=np.complex)

        inci = kx**2 + ky**2 <= hamr**2
        result[:, :, inci] = dipole(kx=kx[inci], ky=ky[inci], **fkwargs)

        # Interpolation out of circle
        outci = np.logical_not(inci)
        result[:, :, outci] = __interpolate(dipole, kx[outci], ky[outci],
                                            b1, b2, hamr, **fkwargs)

        return result


def evaluate_energy(energy, kx, ky, b1, b2, hamr=None,
                    eps=10e-10, **fkwargs):
    """
    Evaluates a function on a given kgrid defined by kx and ky.
    If hamiltonian_radius is given it will interpolate the function
    linearly from the given radius to the edge of the Brillouin zone
    to the other end of the circle.
    """
    if (not np.all(in_brillouin(kx, ky, b1, b2, eps))):
        raise RuntimeError("Error: Not all the given k-points are inside "
                           "the Brillouin zone.")

    if (hamr is None):
        # No hamiltonian region given -> defined in entire bz
        return energy(kx=kx, ky=ky, **fkwargs)
    else:
        # Regular evaluation in circle
        # # Find output shape of the function

        result = np.empty((kx.size, ), dtype=np.complex)

        inci = kx**2 + ky**2 <= hamr**2
        result[inci] = energy(kx=kx[inci], ky=ky[inci], **fkwargs)

        # Interpolation out of circle
        outci = np.logical_not(inci)
        result[outci] = __interpolate(energy, kx[outci], ky[outci],
                                      b1, b2, hamr, **fkwargs)

        return result


def __interpolate(function, kx, ky, b1, b2, hamr, **fkwargs):
    """
    Interpolates everything outside of the Hamiltonian radius with
    a linear function, between the two circle ends
    """
    # Find intersection points with inner hamiltonian circle
    kmat = np.vstack((kx, ky))
    kx_b, ky_b = intersect_brillouin(kx, ky, b1, b2)

    # interpolation length from circle edge over boundary to
    # circle edge
    ipl = 2*(np.linalg.norm(np.vstack((kx_b, ky_b)), axis=0) - hamr)
    knorm = np.linalg.norm(kmat, axis=0)

    # Point at circle, kvector intersection
    kmat_c = hamr*(kmat/knorm)
    pos = knorm - hamr
    # eval_f circle evaluation forward facing to point
    # eval_b circle evaulation on backside facing away from point
    eval_f = function(kx=kmat_c[0], ky=kmat_c[1], **fkwargs)
    eval_b = function(kx=-kmat_c[0], ky=-kmat_c[1], **fkwargs)

    eval_i = (1-pos/ipl)*eval_f + (pos/ipl)*eval_b

    return eval_i
