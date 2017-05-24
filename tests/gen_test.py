from mpi4py import MPI
from neurotrees.io import population_ranges, NeurotreeGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

path = '/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/DGC_forest/hdf5/DGC_forest_20170420.h5'

g = NeurotreeGen(MPI._addressof(comm), path, "GC", io_size=4)#, attributes=True, namespace='Synapse_Attributes')
count = 0
for (i, e) in g:
    print 'Rank %i: gid %i' % (rank, i)
    count = count+1

print 'rank %i: count = %i' % (rank, count)


