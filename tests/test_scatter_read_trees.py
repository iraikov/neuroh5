from mpi4py import MPI
from neurotrees.io import scatter_read_trees, population_ranges

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

pr = population_ranges(MPI._addressof(comm), "data/DGC_forest_test.h5")
(g,_)  = scatter_read_trees(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC", 2, attributes=True)
print "pr = ", pr
for gid in g.keys():
    print "g[%d] = " % gid, g[gid]['Attributes.a']
