from numba import jit
import numba
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import ode
from scipy.integrate import simps


@jit
def econd(k, hop, Delta):
    '''
    Energy of the conduction band. Due to particle-hole symmetry
    the energy of the valence band will be -econd
    '''
    return hop/2 * (np.cos(k) + 1) + Delta/2

@jit
def rabi(omega0, Omega, t):
    '''
    Rabi frequency of the transition.
    '''
    return omega0 * np.sin(Omega * t)

@jit
def diff(x, y):
    if x.size != y.size:
        raise ValueError('Vectors have different lengths.')
    elif y.size == 1:
        return 0
    else:
        return np.gradient(x)/np.gradient(x)

def polarization(p):
    '''
    Calculates the polarization by summing the contribtion from all kpoints.
    '''
    L = p.shape[0]
    return np.real(np.sum(p, axis=0))/L


def current(k, hop, Delta, fc, fv):
    '''
    Calculates the current by summing the contribution from all kpoints.
    '''
    L = np.size(fc, axis=0)
    return np.real(np.dot(diff(k, econd(k, hop, Delta)), fc) + np.dot(diff(k, -econd(k, hop, Delta)), fv + 1))/L

@jit
def f(t, y, k, hop, Delta, gamma1, gamma2, omega0, Omega):
    '''
    Function driving the dynamics of the system.
    This is required as input parameter to the ode solver
    '''

    # Energies of conductance and valence band
    ec = econd(k, hop, Delta)
    ev = -ec

    # Rabi frequency
    wr = rabi(omega0, Omega, t)
    wr_c = np.conjugate(rabi(omega0, Omega, t))

    # Eq. distributions
    fcv = -1

    # Constant added vector
    b = -1j*fcv*np.array([wr, -wr_c, 0 , 0])

    # Main dynamics matrix
    M = -1j*np.array([[(ec - ev) - 1j * gamma2, 0, wr, -wr],\
                [0, -(ec - ev) -1j * gamma2, -wr_c, wr_c],\
                [wr_c, -wr, -1j * gamma1, 0],\
                [-wr_c, wr, 0, -1j * gamma1]])

    # Calculate the timestep
    svec = np.dot(M, y) + b
    return svec

def wrapper(t, y, k, hop, Delta, gamma1, gamma2, omega0, Omega):
    f(t, y, k, hop, Delta, gamma1, gamma2, omega0, Omega)

def main():

    # Parameters
    hop = 1
    Delta = 0.3 * hop
    gamma2 = 0.1 * hop
    gamma1 = gamma2/6
    L = 32
    Omega = 0.5 * hop
    omega0 = 0.01 * hop

    # Initial condition
    y0 = [0.0, 0.0, 0.0, 0.0]
    t0 = 0

    # Final time, time step, time vector
    t1 = 5/gamma1
    dt = 0.1
    t = np.arange(0, t1 + dt, dt)

    # Solutions container
    solution = []

    # Set up solver
    solver = ode(wrapper, jac=None).set_integrator('zvode', method='bdf')

    # Solve for every k in kgrid
    kgrid = np.linspace(-np.pi, np.pi, L)

    for k in kgrid:
        # Append initial condition
        ksolution = []
        ksolution.append(y0)

        # Set solver
        solver.set_initial_value(y0, t0)
        solver.set_f_params(k, hop, Delta, gamma1, gamma2, omega0, Omega)

        while solver.successful() and solver.t + dt <= t1:
            solver.integrate(solver.t + dt)
            ksolution.append(solver.y)

        solution.append(ksolution)

    solution = np.array(solution)

    # Compute polarization and current
    # First index of solution is kpoint, second is timestep, third is pcv, pvc, dfc, dfv
    pol = polarization(solution[:, :, 0])
    curr = current(kgrid, hop, Delta, solution[:, :, 2], solution[:, :, 3]) + diff(t, pol)

    # print(solution[:,:,2])
    # print(solution[:,:,3])

    # Fourier transform
    freq = np.fft.fftfreq(t.size)
    rabifourier = np.fft.fft(rabi(omega0, Omega, t))
    polfourier = np.fft.fft(pol)                               # Polarization
    currfourier = np.fft.fft(curr)                             # Current
    # irad = np.abs(freq*polfourier + 1j*currfourier)**2      # Emission spectrum

    # Average energy per time
#    print(simps(curr * rabi(omega0, Omega, t), t))
#
#    # Plotting goes here
#    # Band structure
#    pl.figure(1)
#    pl.scatter(kgrid, econd(kgrid, hop, Delta), s = 5, label = 'Conduction band')
#    pl.scatter(kgrid, -econd(kgrid, hop, Delta), s = 5, label = 'Valence band')
#    pl.scatter(kgrid, diff(kgrid, econd(kgrid, hop, Delta)), s = 5, label = 'Conduction band velocity')
#    pl.scatter(kgrid, diff(kgrid, -econd(kgrid, hop, Delta)), s = 5, label = 'Valence band velocity')
#
#    ax = pl.gca()
#    ax.set_xlabel(r'$k$', fontsize = 14)
#    ax.legend(loc = 'best')
#
#    pl.figure(2)
##    pl.scatter(t, rabi(omega0, Omega, t), s = 5, label = 'Driving field')
##    pl.scatter(t, pol, s = 5, label = 'Polarization')
##    pl.scatter(t, curr, s = 5, label = 'Current')
#
#    pl.plot(t, rabi(omega0, Omega, t), label = 'Driving field')
#    pl.plot(t, pol, label = 'Polarization')
#    pl.plot(t, curr, label = 'Current')
#
#    ax = pl.gca()
#    ax.set_xlabel(r'$t$', fontsize = 14)
#    ax.legend(loc = 'best')
#
#    # Fourier spectra
#    pl.figure(3)
##    pl.scatter(freq, np.abs(rabifourier), s = 5, label = 'Driving field')
##    pl.scatter(freq, np.abs(polfourier), s = 5,  label = 'Polarization')
##    pl.scatter(freq, np.abs(currfourier), s = 5, label = 'Current')
##    pl.scatter(freq, irad, s = 5, label = 'Emission')
#
#    pl.plot(freq, np.abs(rabifourier), label = 'Driving field')
#    pl.plot(freq, np.abs(polfourier),  label = 'Polarization')
#    pl.plot(freq, np.abs(currfourier), label = 'Current')
#    # pl.plot(freq, irad, label = 'Emission')
#
#    pl.axvline(x = Omega, color = 'k', linestyle = '--')
#
#    ax = pl.gca()
#    ax.set_xlabel(r'$\omega$', fontsize = 14)
#    ax.set_xlim([-0.1, 0.1])
#    ax.legend(loc = 'best')
#
#    pl.show()

if __name__ == "__main__":
    main()
