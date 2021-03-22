from params import params

import cued.hamiltonian
from cued.main import sbe_solver

import numpy as np

def semiconductor():

    a = 6            # lattice constant
    t = 3.0/27.211   # hopping: 1 eV
    m = 3.0/27.211   # on-site energy difference of two sites: 1 eV

    params.length_BZ_E_dir = 2*np.pi/a

    semiconductor_system = cued.hamiltonian.two_site_semiconductor(lattice_const=a, hopping=t, onsite_energy_difference=m)

    return semiconductor_system

def run(system):

    sbe_solver(system, params)

    return 0

if __name__ == "__main__":
    run(semiconductor())
