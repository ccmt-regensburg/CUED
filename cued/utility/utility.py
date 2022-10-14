"""
Utility functions needed by functions/methods in the package
"""
import os
from numba import njit
import numpy as np
import shutil as sh
import sympy as sp
from sympy.utilities.lambdify import lambdify

from cued import CUEDPATH

def cued_copy(cued_dir, dest_dir):
	'''
	Copies files relative to the cued source path
	'''
	sh.copy(CUEDPATH + '/' + cued_dir, dest_dir)


def cued_pwd(cued_file):
	'''
	Returns the full path of a file relative to he cued source path
	'''
	return CUEDPATH + '/' + cued_file


def mkdir(dirname):
	'''
	Only try to create directory when directory does not exist
	'''
	if not os.path.exists(dirname):
		os.mkdir(dirname)

def chdir(dirname='..'):
	'''
	Defaults to go back one folder
	'''
	os.chdir(dirname)


def mkdir_chdir(dirname):
	'''
	Create directory and move into it
	'''
	mkdir(dirname)
	chdir(dirname)


def rmdir_mkdir_chdir(dirname):
	'''
	If the directory exists remove it first before creating
	a new one and changing into it.
	'''
	if os.path.exists(dirname) and os.path.isdir(dirname):
		sh.rmtree(dirname)

	mkdir_chdir(dirname)


class conditional_njit():
	"""
	njit execution only with double precision
	"""
	def __init__(self, precision):
		self.precision = precision

	def __call__(self, func):
		if self.precision in (np.float128, np.complex256):
			return func
		return njit(func)


def matrix_to_njit_functions(sf, hsymbols, dtype=np.complex128, kpflag=False):
	"""
	Converts a sympy matrix into a matrix of functions
	"""
	shp = sf.shape
	jitmat = [[to_njit_function(sf[j, i], hsymbols, dtype, kpflag=kpflag)
			   for i in range(shp[0])] for j in range(shp[1])]
	return jitmat


def list_to_njit_functions(sf, hsymbols, dtype=np.complex128, kpflag=False):
	"""
	Converts a list of sympy functions/matrices to a list of numpy
	callable functions/matrices
	"""
	return [to_njit_function(sfn, hsymbols, dtype, kpflag) for sfn in sf]


def to_njit_function(sf, hsymbols, dtype=np.complex128, kpflag=False):
	"""
	Converts a simple sympy function to a function callable by numpy
	"""

	# Standard k variables
	kx, ky = sp.symbols('kx ky', real=True)

	# Decide wheter we need to use the kp version of the program
	if kpflag:
		kxp, kyp = sp.symbols('kxp kyp', real=True)
		return __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp, dtype=dtype)

	return __to_njit_function_k(sf, hsymbols, kx, ky, dtype=dtype)


def __to_njit_function_k(sf, hsymbols, kx, ky, dtype=np.complex128):
	kset = {kx, ky}
	# Check wheter k is contained in the free symbols
	contains_k = bool(sf.free_symbols.intersection(kset))
	if contains_k:
		# All free Hamiltonian symbols get function parameters
		if dtype == np.complex256:
			return lambdify(list(hsymbols), sf, np)
		return njit(lambdify(list(hsymbols), sf, np))
	# Here we have non k variables in sf. Expand sf by 0*kx*ky
	sf = sf + kx*ky*sp.UnevaluatedExpr(0)
	if dtype == np.complex256:
		return lambdify(list(hsymbols), sf, np)
	return njit(lambdify(list(hsymbols), sf, np))


def __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp, dtype=np.complex128):
	kset = {kx, ky, kxp, kyp}
	hsymbols = hsymbols.union({kxp, kyp})
	# Check wheter k is contained in the free symbols
	contains_k = bool(sf.free_symbols.intersection(kset))
	if contains_k:
		# All free Hamiltonian symbols get function parameters
		if dtype == np.complex256:
			return lambdify(list(hsymbols), sf, np)
		return njit(lambdify(list(hsymbols), sf, np))

	sf = sf + kx*ky*kxp*kyp*sp.UnevaluatedExpr(0)
	if dtype == np.complex256:
		return lambdify(list(hsymbols), sf, np)
	return njit(lambdify(list(hsymbols), sf, np))


def evaluate_njit_matrix(mjit, kx=np.empty(1), ky=np.empty(1), dtype=np.complex128, **fkwargs):
	shp = np.shape(mjit)
	numpy_matrix = np.empty((np.size(kx),) + shp, dtype=dtype)

	for r in range(shp[0]):
		for c in range(shp[1]):
			numpy_matrix[:, r, c] = mjit[r][c](kx=kx, ky=ky, **fkwargs)

	return numpy_matrix
