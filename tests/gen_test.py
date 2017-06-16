from mpi4py import MPI
from neuroh5.io import population_ranges, NeurotreeGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

path = 'data/DGC_forest_test_20170614.h5'

g = NeurotreeGen(MPI._addressof(comm), path, "GC", io_size=2)#, attributes=True, namespace='Synapse_Attributes')
count = 0
for (i, e) in g:
    print 'Rank %i: gid = ' % rank, i
    if not (i is None):
        count = count+1

print 'rank %i: count = %i' % (rank, count)


