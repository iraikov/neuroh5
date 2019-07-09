from mpi4py import MPI
from neuroh5.io import read_trees, write_cell_attributes, append_cell_attributes, read_cell_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

#write_tree_attributes(comm, "data/DGC_forest_test2.h5", "GC", {0: {'a': a, 'b': b}})

a = np.arange(rank*10,(rank+1)*10).astype('uint32')
b = np.arange(rank*20,(rank+1)*20).astype('float32')

print "a = ", a
print "b = ", b
d = {n:{'a': a+n, 'b': b+n} for n in range(rank*5,(rank+1)*5)}
print "rank ",rank,": d.keys = ",d.keys()

pop_name = 'GC'
output_path = "data/write_cell_attr.h5"

if rank == 0:
    attr_dict = d
else:
    attr_dict = {}

write_cell_attributes(output_path, pop_name, attr_dict,
                      namespace='Test Attributes')
