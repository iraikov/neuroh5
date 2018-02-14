import sys

from mpi4py import MPI
from neuroh5.io import bcast_cell_attributes

# import mkl


comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print('%i ranks have been allocated' % comm.size)
sys.stdout.flush()

coords_dir = './data/'
coords_file = 'dentate_Full_Scale_Control_coords_compressed.h5'

soma_coords = {}
source_populations = list(population_ranges(comm, coords_dir+coords_file).keys())
for population in source_populations:
    soma_coords[population] = bcast_cell_attributes(0, coords_dir+coords_file, population,
                                                    namespace='Coordinates')

print(list(soma_coords.keys()))
