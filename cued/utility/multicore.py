import numpy as np
from mpi4py import MPI
import sys

class MpiHelpers:
    '''
    This class holds helper functions for the usage of the
    mpi scatterv and gatherv routines.
    '''
    def __init__(self):
        self.mpi = MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.subcomm = None

    def get_local_idx(self, idxmax):
        # Check whether there are more ranks than indices to partition
        if self.size > idxmax:
            sys.exit("The number of MPI ranks has to be smaller equal "
                     + str(idxmax) + ".")
        # Important mpi.INT == np.int32
        global_idx_list, local_idx_list, ptuple, displace =\
            self.listchop(np.arange(idxmax, dtype=np.int32))
        self.comm.Scatterv([global_idx_list, ptuple, displace, self.mpi.INT], local_idx_list)

        return local_idx_list

    def listchop(self, idxlist):
        if(self.rank == 0):
            ptuple = self.__equipartition(idxlist.size)
            displace = self.__displacelist(ptuple)
        else:
            klist = None
            ptuple = None
            displace = None

        ptuple = self.comm.bcast(ptuple, root=0)
        displace = self.comm.bcast(displace, root=0)

        for i, le in enumerate(ptuple):
            if(self.rank == i):
                idx_local = np.empty(le, dtype=np.int32)
        return idxlist, idx_local, ptuple, displace

    def sync_and_sum(self, local_np_array):
        '''
        sum all data from local processes
        '''

        summed_np_array = np.zeros_like(local_np_array)

        self.subcomm.Barrier()
        self.subcomm.Allreduce(local_np_array, summed_np_array, op=MPI.SUM)
        self.subcomm.Barrier()

        return summed_np_array


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
