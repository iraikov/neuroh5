from mpi4py import MPI
from neuroh5.io import NeuroH5TreeGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

path = 'data/DGC_forest_test_20170623.h5'

g = NeurotreeGen(path, "GC", io_size=2)
count = 0
for (i, e) in g:
    print 'Rank %i: gid = ' % rank, i
    if not (i is None):
        count = count+1

print 'rank %i: count = %i' % (rank, count)


