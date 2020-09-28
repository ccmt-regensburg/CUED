import sympy as sp
import numpy as np

sx = sp.Matrix(((0, 1), (1, 0)))
sy = sp.Matrix(((0, -sp.I), (sp.I, 0)))
sz = sp.Matrix(((1, 0), (0, -1)))

A, R = sp.symbols("A R")
kx, ky = sp.symbols("kx, ky")

H = A * (sx * ky - sy * kx) + 2*R*(kx**3 - 3*kx*ky**2) * sz

print(sp.diff(H, kx))
print(sp.diff(H, ky))
print(H)
