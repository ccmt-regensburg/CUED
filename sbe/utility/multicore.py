import numpy as np
from mpi4py import MPI


class MpiHelpers:
    '''
    This class holds helper functions for the usage of the mpi scatterv and gatherv routines.
    '''
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def listpart(self, krange, *args):
        '''
        Function dividing the given krange into an even list necessary for distribution among
        processors.
        '''
        if(self.rank == 0):
            klist = np.linspace(krange[0], krange[1], krange[2])
            if(args and args[0] == "rand"):
                np.random.shuffle(klist)
            ptuple = self.__equipartition(len(klist))
            displace = self.__displacelist(ptuple)
        else:
            klist = None
            ptuple = None
            displace = None

        ptuple = self.comm.bcast(ptuple, root=0)
        displace = self.comm.bcast(displace, root=0)

        for i, le in enumerate(ptuple):
            if(self.rank == i):
                klocal = np.zeros(le)
        return klist, klocal, ptuple, displace

    def listchop(self, klist):
        if(self.rank == 0):
            ptuple = self.__equipartition(len(klist))
            displace = self.__displacelist(ptuple)
        else:
            klist = None
            ptuple = None
            displace = None

        ptuple = self.comm.bcast(ptuple, root=0)
        displace = self.comm.bcast(displace, root=0)

        for i, le in enumerate(ptuple):
            if(self.rank == i):
                klocal = np.zeros(le)
        return klist, klocal, ptuple, displace

    def __equipartition(self, L):
        '''
        Gives a tuple with entries equal to the number of cores.
        Every entry is an integer giving the amount of elements on the corresponding core
        '''
        div, mod = divmod(L, self.size)
        ptuple = ()
        for i in range(self.size):
            if(i < mod):
                ptuple += (div+1,)
            else:
                ptuple += (div,)
        return ptuple

    def __displacelist(self, ptuple):
        '''
        Gives a tuple with entries equal to the number of cores.
        Every entry is an integer giving the index where to cut the list to be scattered.
        '''
        displace = ()
        for i, p in enumerate(ptuple):
            if(i > 0):
                displace += (displace[i-1] + ptuple[i-1],)
            else:
                displace += (0,)
        return displace
