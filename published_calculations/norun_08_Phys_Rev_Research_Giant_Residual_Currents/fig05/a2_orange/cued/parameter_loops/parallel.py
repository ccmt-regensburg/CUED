import os
from cued.main import sbe_solver


def mkdir(dirname):
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)


def mkdir_chdir(dirname):
    mkdir(dirname)
    os.chdir(dirname)


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


def phasesweep_parallel(phaselist, system, params):
    for phase in phaselist:
        pid = os.fork()

        if pid == 0:
            params.phase = phase
            print("Current phase: ", params.phase)
            dirname_phase = 'phase_{:1.2f}'.format(params.phase)
            mkdir_chdir(dirname_phase)
            sbe_solver(system, params)

            return 0


# def parallel_chirp_phasesweep(chirplist, phaselist, system, dipole, params):

#     for chirp in chirplist:
#         params.chirp = chirp
#         print("Current chirp: ", params.chirp)
#         dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
#         mkdir_chdir(dirname_chirp)

#         for phase in phaselist:
#             params.phase = phase
#             print("Current phase: ", params.phase)
#             dirname_phase = 'phase_{:1.2f}'.format(params.phase)
#             mkdir_chdir(dirname_phase)
#             p = multiprocessing.Process(target=sbe_solver, args=(system, dipole, params,))
#             p.start()
#             p.join()
#             # sbe_solver(system, dipole, params)
#             os.chdir('..')

#         os.chdir('..')

# def mpi_chirp_phasesweep(chirplist, phaselist, system, dipole, params):

#     Multi = MpiHelpers()
#     # Create all chirp folders beforehand
#     if (Multi.rank == 0):
#         for chirp in chirplist:
#             dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
#             mkdir(dirname_chirp)

#     Multi.comm.Barrier()

#     phlist, phlocal, ptuple, displace = Multi.listchop(phaselist)
#     Multi.comm.Scatterv([phlist, ptuple, displace, MPI.DOUBLE], phlocal)

#     for chirp in chirplist:
#         params.chirp = chirp
#         dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
#         print("Rank: ", Multi.rank, " Current chirp: ", params.chirp)
#         os.chdir(dirname_chirp)

#         for phase in phlocal:
#             params.phase = phase
#             print("Rank: ", Multi.rank, " Current phase: ", params.phase)
#             dirname_phase = 'phase_{:1.2f}'.format(params.phase)
#             mkdir_chdir(dirname_phase)
#             sbe_solver(system, dipole, params)
#             os.chdir('..')

#         os.chdir('..')

#     Multi.comm.Barrier()
