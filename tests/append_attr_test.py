from mpi4py import MPI
from neurotrees.io import read_trees, write_tree_attributes, append_tree_attributes, read_tree_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

(g,_) = read_trees(MPI._addressof(comm), "data/DGC_forest_test.h5", "GC")
datasize=3000
a = np.arange(rank*10,(rank+1)*10).astype('uint32')
b = np.arange(rank*20,(rank+1)*20).astype('float32')

#a = np.arange(rank,rank+datasize).astype('uint32')
#b = np.arange(rank+1,rank+1+datasize).astype('uint16')
#c = np.arange(rank+2,rank+2+datasize).astype('float32')
#d = np.arange(rank+3,rank+3+datasize).astype('uint32')
#e = np.arange(rank+4,rank+4+datasize).astype('uint32')

print "a = ", a
print "b = ", b
ranksize=5

#d = {n:{'a': a+n, 'b': b, 'c': c, 'd': d+n, 'e': e+n} for n in g.keys()}
d = {n:{'a': a+n, 'b': b+n} for n in range(rank*ranksize,(rank+1)*ranksize)}

append_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC", d, io_size=2)
append_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC", d, io_size=2)
