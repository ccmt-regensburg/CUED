import numpy as np
import numpy.linalg as lin


class Dipole():
    """
    This class constructs the dipole moments in a phase correct way for a
    given slab Hamiltonian.

    """

    def __init__(self, hamiltonian, kxlist, kylist, gauge_idx=0, **kwargs):
        """
        Constructor initalising the Hamiltonian dimensions and the k-grid

        Parameters
        ----------
        hamiltonian : list
            List of numpy arrays holding the hermitian Hamiltonian.
            h[0] is onsite; h[1], h[2] hop in x; h[3], h[4] hop in y.
        kxlist : np.ndarray
            All kx values the hamiltonian should be diagonalised for.
        kylist : np.ndarray
            All ky values the hamiltonian should be diagonalised for.
        gauge_idx: int
            Index of wave function entry that should be kept real. This
            ultimately determines the gauge of the wave function.

        """
        self.gidx = gauge_idx

        self.h = hamiltonian
        ham_dtype = None
        ham_dim = None

        self.hf = None
        self.hfd = None

        try:
            if callable(self.h) and callable(kwargs['hderivative']):
                self.hf = self.h
                self.hfd = kwargs['hderivative']
                ham_dtype = np.complex
                ham_dim = np.size(self.hf(kxlist[0], kylist[0]), axis=0)
            elif type(self.h) == list or type(self.h) == tuple:
                self.hf = self.__hamiltonian
                self.hfd = self.__hamiltonian_derivative
                ham_dtype = np.max([h.dtype for h in self.h])
                ham_dim = np.size(self.h[0], axis=0)
            else:
                raise ValueError("Error: The Hamiltonian is of the wrong type."
                                 " Should be List/Function.")

        except KeyError:
            raise KeyError("If you define a Hamiltonian function you also have"
                           " to define its derivative.")

        # Find biggest dtype of hamiltonian
        self.kxlist = kxlist
        self.kylist = kylist

        # Containers for energies and wave functions
        self.e = np.empty([kylist.size, kxlist.size,
                          ham_dim], dtype=np.float64)
        self.ev = np.empty([kylist.size, kxlist.size,
                           ham_dim, ham_dim], dtype=np.complex)

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

    def __hamiltonian_derivative(self, kx, ky):
        """
        k-space Hamiltonian derivative if hopping matrices were given.

        """

        ekx = np.exp(1j*kx)
        ekx_d = ekx.conjugate()
        eky = np.exp(1j*ky)
        eky_d = eky.conjugate()
        dxh = 1j*(ekx*self.h[1] - ekx_d*self.h[2])
        dyh = 1j*(eky*self.h[3] - eky_d*self.h[4]),

        return dxh, dyh

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
        that keeps one entry of the wave function fixed as real.

        """

        ev_gauged_entry = np.copy(ev_buff[self.gidx, :])
        ev_buff[self.gidx, :] = np.abs(ev_gauged_entry)
        ev_buff[~(np.arange(np.size(ev_buff, axis=0)) == self.gidx)] *=\
            np.exp(1j*np.angle(ev_gauged_entry.conj()))

    def dipole_field(self, val_idx, con_idx, energy_eps=None):
        """
        Construct the dipole field from the phase correct wave function data.
        This approach uses a direct Hamiltonian derivative.

        Parameters
        ----------
        val_idx : int
            Index of the valence band
        con_idx : int
            Index of the conduction band
        energy_eps: float
            Threshold for the energy differences. This is needed to avoid
            divergencies in the dipole field and replace them with 'np.nan'

        Returns
        -------
        A : np.ndarray
            x-component of dipole field. Rows are ky, Columns kx and Depth is
            Ax, Ay.

        """

        A = np.empty([self.kylist.size, self.kxlist.size, 2], dtype=np.complex)
        dxwf = np.empty(self.ev[:, :, val_idx].shape, dtype=np.complex)
        dywf = np.empty(self.ev[:, :, val_idx].shape, dtype=np.complex)

        for i, ky in enumerate(self.kylist):
            for j, kx in enumerate(self.kxlist):
                dxh, dyh = self.hfd(kx, ky)
                dxwf[i, j] = np.dot(dxh, self.ev[i, j, :, val_idx])
                dywf[i, j] = np.dot(dyh, self.ev[i, j, :, val_idx])

        A[:, :, 0] = 1j*np.einsum('ijk,ijk->ij',
                                  self.ev[:, :, :, con_idx].conjugate(), dxwf)
        A[:, :, 1] = 1j*np.einsum('ijk,ijk->ij',
                                  self.ev[:, :, :, con_idx].conjugate(), dywf)
        delta_E = self.e[:, :, val_idx] - self.e[:, :, con_idx]

        # Do not divide by numbers that are too small
        if energy_eps != None:
            delta_E[np.abs(delta_E) < energy_eps] = np.nan


        A[:, :, 0] /= delta_E
        A[:, :, 1] /= delta_E

        self.dipole_fields.update({(val_idx, con_idx): A})

        return A
    
    def dipole_field_deriv(self, val_idx, con_idx, energy_eps=None):
        """
        Construct the dipole field from the phase correct wave function data.
        This approach uses a direct Hamiltonian derivative.

        Parameters
        ----------
        val_idx : int
            Index of the valence band
        con_idx : int
            Index of the conduction band
        energy_eps: float
            Threshold for the energy differences. This is needed to avoid
            divergencies in the dipole field and replace them with 'np.nan'

        Returns
        -------
        A : np.ndarray
            x-component of dipole field. Rows are ky, Columns kx and Depth is
            Ax, Ay.

        """

        A = np.empty([self.kylist.size, self.kxlist.size, 2], dtype=np.complex)
        dxwf = np.empty(self.ev[:, :, val_idx].shape, dtype=np.complex)
        dywf = np.empty(self.ev[:, :, val_idx].shape, dtype=np.complex)

        for i, ky in enumerate(self.kylist):
            for j, kx in enumerate(self.kxlist):
                dxh, dyh = self.hfd(kx, ky)
                dxwf[i, j] = np.dot(dxh, self.ev[i, j, :, val_idx])
                dywf[i, j] = np.dot(dyh, self.ev[i, j, :, val_idx])

        A[:, :, 0] = 1j*np.einsum('ijk,ijk->ij',
                                  self.ev[:, :, :, con_idx].conjugate(), dxwf)
        A[:, :, 1] = 1j*np.einsum('ijk,ijk->ij',
                                  self.ev[:, :, :, con_idx].conjugate(), dywf)
        delta_E = self.e[:, :, val_idx] - self.e[:, :, con_idx]

        # Do not divide by numbers that are too small
        if energy_eps != None:
            delta_E[np.abs(delta_E) < energy_eps] = np.nan


        A[:, :, 0] /= delta_E
        A[:, :, 1] /= delta_E

        self.dipole_fields.update({(val_idx, con_idx): A})

        return A
