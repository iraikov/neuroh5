from mpi4py import MPI
from neuroh5.io import population_ranges, NeuroH5CellAttrGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

g = NeuroH5CellAttrGen(comm, "data/DGC_forest_attr_test_20170614.h5", "GC", io_size=comm.size)

for (i, e) in g:
    if i is not None:
        print 'rank %i: gid = %i' % (rank, i)
#    print (i, e)

