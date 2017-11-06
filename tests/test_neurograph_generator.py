import sys,gc
from mpi4py import MPI
from neuroh5.io import NeuroH5ProjectionGen, read_projection_names
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

path = './data/dentate_test_connectivity.h5'
path = '/home/igr/src/model/dentate/datasets/DG_test_connections_20171022.h5'

for (src,dst) in read_projection_names(comm, path):
    print 'Projection %s -> %s' % (src, dst)
    g = NeuroH5ProjectionGen (comm, path, src, dst, namespaces=['Synapses'])
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

