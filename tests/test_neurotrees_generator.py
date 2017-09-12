from mpi4py import MPI
from neuroh5.io import read_population_ranges, append_cell_attributes, bcast_cell_attributes, NeurotreeGen, NeurotreeAttrGen

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

neurotrees_dir = 'data/'
# neurotrees_dir = os.environ['PI_SCRATCH']+'/DGC_forest/hdf5/'
# neurotrees_dir = os.environ['PI_HOME']+'/'
forest_file = 'DGC_forest_connectivity_040617.h5'
test_file = 'DGC_forest_syns_test_012717.h5'

# synapse_dict = read_from_pkl(neurotrees_dir+'010117_GC_test_synapse_attrs.pkl')
#synapse_dict = read_tree_attributes(MPI._addressof(comm), neurotrees_dir+forest_file, 'GC',
#                                    namespace='Synapse_Attributes')

coords_dir = 'data/'
# coords_dir = os.environ['PI_SCRATCH']+'/DG/'
# coords_dir = os.environ['PI_HOME']+'/Full_Scale_Control/'
coords_file = 'dentate_Full_Scale_Control_coords_compressed.h5'


g = NeurotreeAttrGen(comm, neurotrees_dir+test_file, 'GC', io_size=comm.size,
                     namespace='Synapse_Attributes')
global_count = 0
count = 0
for target_gid, synapse_dict in g:
    print 'Rank: %i, gid: %i, count: %i' % (rank, target_gid, count)
    count += 1
global_count = comm.gather(count, root=0)
if rank == 0:
    print 'Total: %i' % np.sum(global_count)

# test = population_ranges(MPI._addressof(comm), coords_dir+coords_file)
test = bcast_cell_attributes(comm, 0, coords_dir+coords_file, 'GC', namespace='Coordinates')
