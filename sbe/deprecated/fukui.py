import numpy as np
import numpy.linalg as lin


class Dipole():
    """
    This class constructs the dipole moments in a phase correct way for a
    given slab Hamiltonian.
    """

    def __init__(self, hamiltonian, kxlist, kylist, gidx=0):
        """
        Constructor initalising the Hamiltonian dimensions and the k-grid

        Parameters
        ----------
        hamiltonian : list
            List of numpy arrays holding the hermitian Hamiltonian.
            h[0] is onsite; h[1], h[2] hop in x; h[3], h[4] hop in y
        kxlist : np.ndarray
            All kx values the hamiltonian should be diagonalised for
        kylist : np.ndarray
            All ky values the hamiltonian should be diagonalised for
        gidx: int
            Index of wave function entry that should be kept real. This
            ultimately determines the gauge of the wave function
        """
        self.gidx = gidx

        self.h = hamiltonian
        ham_dtype = None
        ham_dim = None

        self.hf = None

        if callable(self.h):
            self.hf = self.h
            ham_dtype = np.complex
            ham_dim = np.size(self.hf(kxlist[0], kylist[0]), axis=0)
        elif type(self.h) == list or type(self.h) == tuple:
            self.hf = self.__hamiltonian
            ham_dtype = np.max([h.dtype for h in self.h])
            ham_dim = np.size(self.h[0], axis=0)
        else:
            raise ValueError("Error: The Hamiltonian is of the wrong type."
                             " Should be List/Function.")

        # Find biggest dtype of hamiltonian
        self.kxlist = kxlist
        self.kylist = kylist

        # Containers for energies and wave functions
        self.e = np.empty([kylist.size, kxlist.size,
                          ham_dim], dtype=np.float64)
        self.ev = np.empty([kylist.size, kxlist.size,
                           ham_dim, ham_dim], dtype=ham_dtype)

        self.__diagonalise()

        self.dipole_fields = dict()

    def __hamiltonian(self, kx, ky):
        """
        Form of the k-space Hamiltonian if hopping matrices are given.
        """

        ekx = np.exp(1j*kx)
        ekx_d = ekx.conjugate()
        eky = np.exp(1j*ky)
        eky_d = eky.conjugate()
        return self.h[0]\
             + self.h[1]*ekx + self.h[2]*ekx_d\
             + self.h[3]*eky + self.h[4]*eky_d

    def __diagonalise(self):
        """
        Diagonalise the full hamiltonian for every k-point and order the wave
        functions accordingly. Also take care of the correct phase relation
        between wave functions via a gauge transform (first entry real)
        """

        for i, ky in enumerate(self.kylist):
            for j, kx in enumerate(self.kxlist):
                self.e[i, j], ev_buff = lin.eigh(self.hf(kx, ky))
                # Gauge transform for correct phase relations
                # Fixed to the index given at instance __init__
                self.__gauge(ev_buff)

                self.ev[i, j] = ev_buff

    def __gauge(self, ev_buff):
        """
        Applies a gauge transformation to the wave functions in order to make
        their relatives phases correct. This is a simple gauge transformation
        that keeps one entry of the wave function fixex as real.
        """

        ev_gauged_entry = np.copy(ev_buff[self.gidx, :])
        ev_buff[self.gidx, :] = np.abs(ev_gauged_entry)
        ev_buff[~(np.arange(np.size(ev_buff, axis=0)) == self.gidx)] *=\
            np.exp(1j*np.angle(ev_gauged_entry.conj()))

    def dipole_field(self, val_idx, con_idx):
        """
        Construct the dipole field from the phase correct wave function data.
        This approach uses a direct Hamiltonian derivative.

        Parameters
        ----------
        val_idx : int
            Index of the valence band
        con_idx : int
            Index of the conduction band

        Returns
        -------
        A : np.ndarray
            x-component of dipole field. Rows are ky, Columns kx and Depth is
            Ax, Ay.
        """
        A = np.empty([self.kxlist.size, self.kylist.size, 2], dtype=np.complex)
        kx_shift = np.arange(-1, self.kxlist.size-1)
        ky_shift = np.arange(-1, self.kylist.size-1)

        # u10 component of fukui link variable (kx-component)
        u10 = np.einsum("ijk,ijk->ij",
                        self.ev[:, :, :, val_idx].conj(),
                        self.ev[:, kx_shift, :, val_idx], optimize=True)
        u10 /= np.abs(u10)
        # u20 component of fukui link variable (ky-component)
        u20 = np.einsum("ijk,ijk->ij",
                        self.ev[:, :, :, val_idx].conj(),
                        self.ev[ky_shift, :, :, val_idx], optimize=True)
        u20 /= np.abs(u20)
        A[:, :, 0] = np.log(u10)
        A[:, :, 1] = np.log(u20)


        # # Comments on Berry Curvature
        # u12 = np.einsum("ijk,ijk->ij", self.ev[:-1, 1:, :, con_idx].conj(),
        #                 self.ev[1:, 1:, :, val_idx], optimize=True)
        # u21 = np.einsum("ijk,ijk->ij", self.ev[1:, :-1, :, con_idx].conj(),
        #                 self.ev[1:, 1:, :, val_idx], optimize=True)

        # f12 = np.log(u10*u21*(1.0/u12)*(1.0/u20))

        # return np.sum(f12)/(2j*np.pi)

        return A
