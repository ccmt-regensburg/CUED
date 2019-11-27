from sympy import *


class SymbolicDipole():
    """
    This class constructs the dipole moment functions from a given symbolic
    Hamiltonian and wave function. It also performs checks on the input
    wave function to guarantee orthogonality and normalisation.

    """

    def __init__(self, h, e, wf):
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


    def __test_input(self, h, e, wf):
        

