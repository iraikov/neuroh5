from mpi4py import MPI
from neuroh5.io import scatter_read_trees, read_population_ranges

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

(g,n)  = scatter_read_trees(comm, "data/DGC_forest_test_attrs.h5", "GC", io_size=2)

for gid in g.keys():
    print "g[%d] = " % gid, g[gid]
