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


def in_brillouin(kx, ky, b1, b2, eps):
    """
    Check if a collection of k-points is inse a zone determined
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

    return kx[is_inzone], ky[is_inzone]


def intersect_brillouin(kx, ky, b1, b2):
    """
    Find the intersection points of a line through a point (kx, ky)
    with the brillouin zone boundary
    """
    a1, a2 = to_reciprocal_coordinates(kx, ky, b1, b2)
    beta = np.array([0.5/a1, 0.5/a2])
    beta = np.min(np.abs(beta), axis=0)
    a1_b, a2_b = beta*a1, beta*a2
    return to_cartesian_coordinates(a1_b, a2_b, b1, b2)
