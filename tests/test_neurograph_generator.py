import sys,gc
from mpi4py import MPI
from neuroh5.io import NeuroH5ProjectionGen, read_projection_names
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

for (src,dst) in read_projection_names(comm, 'data/dentate_test.h5'):
    g = NeuroH5ProjectionGen (comm, 'data/dentate_test.h5', src, dst)
    print g
    for (i,rest) in g:
        if i is not None:
            (adj,attr) = rest
            print i, adj, attr

#g = NeuroH5ProjectionGen (comm, 'data/dentate_test.h5', 'BC', 'MC')
#for (i,j,attr) in g:
#        if i is not None:
#            print i, j
gc.collect()

