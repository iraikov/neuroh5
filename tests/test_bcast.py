import sys
from mpi4py import MPI
from neurotrees.io import append_cell_attributes
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import bcast_cell_attributes
from neurotrees.io import population_ranges
# import mkl


comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

coords_dir = './data/'
coords_file = 'dentate_Full_Scale_Control_coords_compressed.h5'

soma_coords = {}
source_populations = population_ranges(MPI._addressof(comm), coords_dir+coords_file).keys()
for population in source_populations:
    soma_coords[population] = bcast_cell_attributes(MPI._addressof(comm), 0, coords_dir+coords_file, population,
                                                    namespace='Coordinates')

print soma_coords.keys()
