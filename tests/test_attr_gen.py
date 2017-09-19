from mpi4py import MPI
from neuroh5.io import read_population_ranges, NeuroH5CellAttrGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank


if rank == 0:
    print read_population_ranges(comm, "data/dentate_Sampled_Soma_Locations.h5")
g = NeuroH5CellAttrGen(comm, "data/dentate_Sampled_Soma_Locations.h5", "MOPP", io_size=comm.size, namespace='Coordinates')

for (i, e) in g:
    print i, e

