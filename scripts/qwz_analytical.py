import sympy as sp

import hfsbe.example as ex

m = sp.Symbol('m')
kx = sp.Symbol('kx')
ky = sp.Symbol('ky')

h, e, wf = ex.TwoBandSystems(no_norm=True).qwz()

vv = wf[0][0, :]
sp.pprint(sp.simplify(sp.diff(vv[1], kx)))
