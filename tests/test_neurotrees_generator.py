from mpi4py import MPI
from neuroh5.io import read_population_ranges, NeuroH5TreeGen, NeuroH5CellAttrGen

# import mkl
import sys
import os
import gc
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

neurotrees_dir = os.environ['SCRATCH']+'/dentate/Full_Scale_Control/'
#forest_file = 'DGC_forest_test_syns_20171019.h5'
forest_file = 'DGC_forest_syns_compressed_20180306.h5'

g = NeuroH5CellAttrGen(neurotrees_dir+forest_file, 'GC', comm=comm, io_size=160,
                     namespace='Synapse Attributes')
global_count = 0
count = 0
for destination_gid, synapse_dict in g:
    if destination_gid is None:
        print ('Rank %i destination gid is None' % rank)
    else:
        print 'Rank: %i, gid: %i, count: %i' % (rank, destination_gid, count)
        count += 1
global_count = comm.gather(count, root=0)
if rank == 0:
    print 'Total: %i' % np.sum(global_count)


