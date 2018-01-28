from mpi4py import MPI
from neuroh5.io import scatter_read_trees, read_population_ranges, read_cell_attribute_info

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

ranges = read_population_ranges ("data/DGC_forest_test_attrs.h5")
print ranges

attribute_info = read_cell_attribute_info (["GC"], "data/DGC_forest_test_attrs.h5")
print attribute_info

(g,n)  = scatter_read_trees("data/DGC_forest_test_attrs.h5", "GC", io_size=2)

for (gid, tree) in g:
    print "g[%d] = " % gid, tree
