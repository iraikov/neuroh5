from mpi4py import MPI
from neurotrees.io import population_ranges, NeurotreeAttrGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

#print population_ranges(MPI._addressof(comm), "data/dentate_Full_Scale_Control_coords_compressed.h5")
if rank == 0:
    print population_ranges(MPI._addressof(comm), "data/dentate_Sampled_Soma_Locations.h5")
#g = NeurotreeAttrGen(MPI._addressof(comm), "data/dentate_Full_Scale_Control_coords_compressed.h5", "GC", io_size=2, namespace='Coordinates')
#g = NeurotreeAttrGen(MPI._addressof(comm), "data/DGC_forest_syn_locs_test_20170421.h5", "GC", io_size=comm.size, namespace='Synapse_Attributes')
g = NeurotreeAttrGen(MPI._addressof(comm), "data/dentate_Sampled_Soma_Locations.h5", "MOPP", io_size=comm.size, namespace='Coordinates')

for (i, e) in g:
    print i, e

