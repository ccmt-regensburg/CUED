import numpy as np


def to_reciprocal_coordinates(kx, ky, b1, b2):
    kmat = np.vstack((kx, ky))
    a1 = b1.dot(kmat)/(np.linalg.norm(b1)**2)
    a2 = b2.dot(kmat)/(np.linalg.norm(b2)**2)
    return a1, a2


def to_cartesian_coordinates(a1, a2, b1, b2):
    kmat = a1*b1[:, np.newaxis] + a2*b2[:, np.newaxis]
    kx = kmat[0]
    ky = kmat[1]
    return kx, ky
