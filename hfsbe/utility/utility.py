"""
Utility functions needed by functions/methods in the package
"""
import sympy as sp


def to_numpy_function(sf):
    """
    Converts a simple sympy function/matrix to a function/matrix
    callable by numpy
    """
    symbols = sf.free_symbols

    # This is to readd kx and ky if they got removed in the
    # simplification process. The variable will just return 0's
    # if used.

    symbols.update([sp.Symbol('kx'), sp.Symbol('ky')])
    return sp.lambdify(symbols, sf, "numpy")


def list_to_numpy_functions(sf):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_numpy_function(sfn) for sfn in sf]
