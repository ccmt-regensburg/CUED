"""
Utility functions needed by functions/methods in the package
"""
from numba import njit
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


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

    # Decide wheter we need to use the kp version of the program
    if kpflag:
        kxp, kyp = sp.symbols('kxp kyp', real=True)
        return __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp)

    return __to_njit_function_k(sf, hsymbols, kx, ky)


def __to_njit_function_k(sf, hsymbols, kx, ky):
    kset = {kx, ky}
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if contains_k:
        # All free Hamiltonian symbols get function parameters
        return njit(lambdify(hsymbols, sf, "numpy"))

    # Here we have non k variables in sf. Expand sf by 0*kx*ky
    sf = sf + kx*ky*sp.UnevaluatedExpr(0)
    return njit(lambdify(hsymbols, sf, "numpy"))


def __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp):
    kset = {kx, ky, kxp, kyp}
    hsymbols = hsymbols.union({kxp, kyp})
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if contains_k:
        # All free Hamiltonian symbols get function parameters
        return njit(lambdify(hsymbols, sf, "numpy"))

    sf = sf + kx*ky*kxp*kyp*sp.UnevaluatedExpr(0)
    return njit(lambdify(hsymbols, sf, "numpy"))


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

    # This is to read kx and ky if they got removed in the
    # simplification process. The variable will just return 0's
    # if used.
    if kx in symbols and ky in symbols:
        return lambdify(symbols, sf, "numpy")

    if (kx not in symbols and ky in symbols):
        symbols.add(kx)
        return lambdify(symbols, sf, "numpy")
    if (kx in symbols and ky not in symbols):
        symbols.add(ky)
        return lambdify(symbols, sf, "numpy")

    symbols.update([kx, ky])
    func = lambdify(symbols, sf, "numpy")

    def __func(kx=kx, ky=ky, **fkwargs):
        dim = kx.size
        out = func(kx, ky, **fkwargs)
        return np.zeros(np.shape(out) + (dim,))

    return __func
