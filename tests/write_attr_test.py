from mpi4py import MPI
from neurotrees.io import read_trees, write_tree_attributes, append_tree_attributes, read_tree_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

g = read_trees(MPI._addressof(comm), "data/DGC_forest_test.h5", "GC")
#write_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test2.h5", "GC", {0: {'a': a, 'b': b}})

a = np.arange(rank*10,(rank+1)*10).astype('uint32')
b = np.arange(rank*20,(rank+1)*20).astype('float32')
print "a = ", a
print "b = ", b
d = {n:{'a': a+n, 'b': b+n} for n in range(rank*5,(rank+1)*5)}
print "rank ",rank,": d.keys = ",d.keys()
write_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test2.h5", "GC", d)



