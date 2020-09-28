import numpy as np

def dipole_field(kpos):
    A = 2.8413
    R = -3.3765
    kx = kpos[:, 0]
    ky = kpos[:, 1]

    Acv = np.zeros(kpos.shape, dtype=np.complex)

    Acv[:, 0] = (0.5j*A*(A**2*(kx - 1j*ky)*ky + \
        2*(2j*kx**3 + 3*kx**2*ky - 3*ky**3)*R* \
        (2*kx**3*R - 6*kx*ky**2*R + np.sqrt(A**2*(kx**2 + ky**2) + \
        4*kx**2*(kx**2 - 3*ky**2)**2*R**2))))/ \
        ((A**2*(kx**2 + ky**2) + 4*(kx**3 - 3*kx*ky**2)**2*R**2)* \
        (2*(kx**3 - 3*kx*ky**2)*R + np.sqrt(A**2*(kx**2 + ky**2) + \
        4*(kx**3 - 3*kx*ky**2)**2*R**2)))

    Acv[:, 1] = (-0.5j*A*kx*(A**2*(kx - 1j*ky) + \
        2*(kx**2 + 6j*kx*ky + 3*ky**2)*R* \
        (2*kx**3*R - 6*kx*ky**2*R + \
        np.sqrt(A**2*(kx**2 + ky**2) + \
        4*kx**2*(kx**2 - 3*ky**2)**2*R**2))))/ \
        ((A**2*(kx**2 + ky**2) + 4*(kx**3 - 3*kx*ky**2)**2*R**2)* \
        (2*(kx**3 - 3*kx*ky**2)*R + \
        np.sqrt(A**2*(kx**2 + ky**2) + 4*(kx**3 - \
        3*kx*ky**2)**2*R**2)))

    return Acv, Acv.conjugate()

def berry_connection(kpos):
    A = 2.8413
    R = -3.3765

    Avv = np.zeros(kpos.shape, dtype=np.complex)
    Acc = np.zeros(kpos.shape, dtype=np.complex)
    
if __name__ == "__main__":
    kpos = np.array([[1, 1], [1, 0.5], [1, 0]])
    Acv = dipole_field(kpos)
    print(Acv)
