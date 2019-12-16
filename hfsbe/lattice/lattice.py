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
    beta = np.array([0.5/a1, 0.5/a2])
    beta = np.min(np.abs(beta), axis=0)
    a1_b, a2_b = beta*a1, beta*a2
    return to_cartesian_coordinates(a1_b, a2_b, b1, b2)


def evaluate_in_brillouin(function, kx, ky, b1, b2,
                          hamiltonian_radius=None, eps=10e-10,
                          **fkwargs):
    """
    Evaluates a function on a given kgrid defined by kx and ky.
    If hamiltonian_radius is given it will interpolate the function
    linearly from the given radius to the edge of the Brillouin zone
    to the other end of the circle.
    """
    hamr = hamiltonian_radius
    if (not np.all(in_brillouin(kx, ky, b1, b2, eps)):
        raise Runtime("Error: Not all the given k-points are inside
                      the Brillouin zone.")

    if (hamr is None):
        # No hamiltonian region given -> defined in entire bz
        return function(kx=kx, ky=ky, **fkwargs)
    else:
        minlen = np.min(np.linalg.norm(self.b1),
                        np.linalg.norm(self.b2))
        hamr *= minlen
        # Regular evaluation in circle
        Ax = np.empty(self.Ax.shape + (kx.size, ), dtype=np.complex)
        Ay = np.empty(self.Ay.shape + (ky.size, ), dtype=np.complex)

        inci = kx**2 + ky**2 <= hamr**2
        Ax[:, :, inci] = self.Axf(kx=kx[inci], ky=ky[inci],
                                  **self.fkwargs)
        Ay[:, :, inci] = self.Ayf(kx=kx[inci], ky=ky[inci],
                                  **self.fkwargs)

        # Interpolation out of circle
        outci = np.logical_not(inci)
        Axi, Ayi = self.__interpolate(kx[outci], ky[outci], hamr)
        Ax[:, :, outci] = Axi
        Ay[:, :, outci] = Ayi

        return kxbz, kybz, Ax, Ay

