import os
from cued.main import sbe_solver
from cued.utility import mkdir_chdir


def chirp_phasesweep(chirplist, phaselist, system, params):

    for chirp in chirplist:
        params.chirp = chirp
        print("Current chirp: ", params.chirp)
        dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
        mkdir_chdir(dirname_chirp)

        for phase in phaselist:
            params.phase = phase
            print("Current phase: ", params.phase)
            dirname_phase = 'phase_{:1.2f}'.format(params.phase)
            mkdir_chdir(dirname_phase)
            sbe_solver(system, params)
            os.chdir('..')

        os.chdir('..')
