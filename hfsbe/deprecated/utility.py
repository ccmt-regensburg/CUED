"""
Utility functions needed by functions/methods in the package
"""
from numba import njit
import numpy as np
import sympy as sp


def matrix_to_njit_functions(sf):
    """
    Converts a sympy matrix into a matrix of functions
    """
    shp = sf.shape
    jitmat = [[to_njit_function(sf[j, i]) for i in range(shp[0])]
              for j in range(shp[1])]
    return jitmat


def list_to_njit_functions(sf):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_njit_function(sfn) for sfn in sf]


def to_njit_function(sf):
    """
    Converts a simple sympy function to a function callable by numpy
    """
    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)
    symbols = sf.free_symbols

    # This is to readd kx and ky if they got removed in the
    # simplification process. The variable will just return 0's
    # if used.
    if (kx in symbols and ky in symbols):
        return njit(sp.lambdify(symbols, sf, "numpy"))
    else:
        if (kx not in symbols and ky in symbols):
            symbols.add(kx)
            return njit(sp.lambdify(symbols, sf, "numpy"))
        if (kx in symbols and ky not in symbols):
            symbols.add(ky)
            return njit(sp.lambdify(symbols, sf, "numpy"))

        prefac = complex(sf)

        def __func(kx=np.empty(1), ky=np.empty(1), **fkwargs):
            dim = kx.size
            return np.repeat(prefac, dim)

        return njit(__func)


def evaluate_njit_matrix(mjit, kx=np.empty(1), ky=np.empty(1)):
    shp = np.shape(mjit)
    numpy_matrix = np.empty(shp + (kx.size,), dtype=np.complex)

    for r in range(shp[0]):
        for c in range(shp[1]):
            numpy_matrix[r, c] = mjit[r][c](kx=kx, ky=ky)

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
