"""
Utility functions needed by functions/methods in the package
"""
from numba import njit
import numpy as np
import sympy as sp


def matrix_to_njit_functions(sf, hsymbols, kpflag=False):
    """
    Converts a sympy matrix into a matrix of functions
    """
    shp = sf.shape
    jitmat = [[to_njit_function(sf[j, i], hsymbols, kpflag=kpflag)
               for i in range(shp[0])] for j in range(shp[1])]
    return jitmat


def list_to_njit_functions(sf, hsymbols, kpflag=False):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_njit_function(sfn, hsymbols, kpflag) for sfn in sf]


def to_njit_function(sf, hsymbols, kpflag=False):
    """
    Converts a simple sympy function to a function callable by numpy
    """

    # Standard k variables
    kx, ky = sp.symbols('kx ky', real=True)
    kset = {kx, ky}

    # Decide wheter we need to use the kp version of the program
    if (kpflag):
        kxp, kyp = sp.symbols('kxp kyp', real=True)
        kpset = {kxp, kyp}
        return __to_njit_function_kp(sf, hsymbols, kset, kpset)
    else:
        return __to_njit_function_k(sf, hsymbols, kset)


def __to_njit_function_k(sf, hsymbols, kset):
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if (contains_k):
        # All free Hamiltonian symbols get function parameters
        return njit(sp.lambdify(hsymbols, sf, "numpy"))
    if (bool(sf.free_symbols)):
        # Here we have non k variables in sf. That's why we will lambdify
        # those and make a k-dependent wrapper for repeating
        sfunc = njit(sp.lambdify(hsymbols.difference(kset), sf, "numpy"))

        def __func(kx=np.empty(1), ky=np.empty(1), **fkwargs):
            dim = kx.size
            return np.repeat(sfunc(**fkwargs), dim)

        return njit(__func)
    else:
        # If we are here sf.free_symbols does not contain any variable
        # Prepare dummy function only repeating the constant
        prefac = complex(sf)

        def __func(kx=np.empty(1), ky=np.empty(1), **fkwargs):
            dim = kx.size
            return np.repeat(prefac, dim)

        return njit(__func)


def __to_njit_function_kp(sf, hsymbols, kset, kpset):
    kset = kset.union(kpset)
    hsymbols = hsymbols.union(kpset)
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if (contains_k):
        # All free Hamiltonian symbols get function parameters
        return njit(sp.lambdify(hsymbols, sf, "numpy"))
    if (bool(sf.free_symbols)):
        # Here we have non k variables in sf. That's why we will lambdify
        # those and make a k-dependent wrapper for repeating
        sfunc = njit(sp.lambdify(hsymbols.difference(kset), sf, "numpy"))

        def __func(kx=np.empty(1), ky=np.empty(1),
                   kxp=np.empty(1), kyp=np.empty(1), **fkwargs):
            dim = kxp.size
            return np.repeat(sfunc(**fkwargs), dim)

        return njit(__func)
    else:
        # If we are here sf.free_symbols does not contain any variable
        # Prepare dummy function only repeating the constant
        prefac = complex(sf)

        def __func(kx=np.empty(1), ky=np.empty(1),
                   kxp=np.empty(1), kyp=np.empty(1), **fkwargs):
            dim = kxp.size
            return np.repeat(prefac, dim)

        return njit(__func)


def evaluate_njit_matrix(mjit, kx=np.empty(1), ky=np.empty(1),
                         **fkwargs):
    shp = np.shape(mjit)
    numpy_matrix = np.empty(shp + (kx.size,), dtype=np.complex)

    for r in range(shp[0]):
        for c in range(shp[1]):
            numpy_matrix[r, c] = mjit[r][c](kx=kx, ky=ky, **fkwargs)

    return numpy_matrix


def list_to_numpy_functions(sf):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_numpy_function(sfn) for sfn in sf]


def to_numpy_function(sf):
    """
    Converts a simple sympy function/matrix to a function/matrix
    callable by numpy
    """
    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)
    symbols = sf.free_symbols

    # This is to readd kx and ky if they got removed in the
    # simplification process. The variable will just return 0's
    # if used.
    if (kx in symbols and ky in symbols):
        return sp.lambdify(symbols, sf, "numpy")
    else:
        if (kx not in symbols and ky in symbols):
            symbols.add(kx)
            return sp.lambdify(symbols, sf, "numpy")
        if (kx in symbols and ky not in symbols):
            symbols.add(ky)
            return sp.lambdify(symbols, sf, "numpy")

        symbols.update([kx, ky])
        func = sp.lambdify(symbols, sf, "numpy")

        def __func(kx=kx, ky=ky, **fkwargs):
            dim = kx.size
            out = func(kx, ky, **fkwargs)
            return np.zeros(np.shape(out) + (dim,))

        return __func
