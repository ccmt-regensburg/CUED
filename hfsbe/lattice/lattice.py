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
