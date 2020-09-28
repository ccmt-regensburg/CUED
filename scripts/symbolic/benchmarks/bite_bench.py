import numpy as np

from hfsbe.example import BiTe
from hfsbe.dipole import SymbolicDipole, SymbolicCurvature

if __name__ == "__main__":
    N = 400
    tsteps = 2000

    bite = BiTe(A=0.1974, R=11.06, C0=0, C2=0, kcut=0.05)
    h, e, wf, ed = bite.eigensystem(gidx=0)

    dip = SymbolicDipole(h, e, wf)
    cur = SymbolicCurvature(dip.Ax, dip.Ay)

    for i in range(tsteps):
        print("Step " + str(i+1) + "/" + str(tsteps))
        kx = np.random.random_sample(N)
        ky = np.random.random_sample(N)

        bite.evaluate_energy(kx, ky)
        bite.evaluate_ederivative(kx, ky)
        dip.evaluate(kx, ky)
        cur.evaluate(kx, ky)
