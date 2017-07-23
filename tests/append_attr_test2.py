from mpi4py import MPI
from neurotrees.io import read_trees, write_tree_attributes, append_tree_attributes, read_tree_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

(g,_) = read_trees(MPI._addressof(comm), "data/DGC_forest_test.h5", "GC")
datasize=3000
a = np.arange(0,datasize).astype('uint32')
b = np.arange(1,1+datasize).astype('uint16')
c = np.arange(2,2+datasize).astype('float32')
d = np.arange(3,3+datasize).astype('uint32')
e = np.arange(4,4+datasize).astype('uint32')
print a.dtype
ranksize=100
ks = g.keys()
d1 = {n:{'a': a+n, 'b': b, 'c': c, 'd': d+n, 'e': e+n} for n in ks[0:len(ks)/2]}
d2 = {n:{'a': a+n, 'b': b, 'c': c, 'd': d+n, 'e': e+n} for n in ks[len(ks)/2:]}

print "rank ",rank,": d1.keys = ",d1.keys()
print "rank ",rank,": d2.keys = ",d2.keys()
append_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC", d1, io_size=2)
append_tree_attributes(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC", d2, io_size=2)
