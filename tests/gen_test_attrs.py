from mpi4py import MPI
from neurotrees.io import population_ranges, NeurotreeGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

g = NeurotreeGen(MPI._addressof(comm), "data/DGC_forest_syns_test_012717.h5", "GC", io_size=comm.size, attributes=True, namespace='Synapse_Attributes')

for (i, e) in g:
    print 'rank %i: gid = %i' % (rank, i)
#    print (i, e)

