from mpi4py import MPI
from neuroh5.io import read_trees, write_cell_attributes, append_cell_attributes, read_cell_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

(g,_) = read_trees("data/DGC_forest_append_test_20180116.h5", "GC")
datasize=3000
a = np.arange(rank*10,(rank+1)*10).astype('uint32')
b = np.arange(rank*20,(rank+1)*20).astype('float32')

#a = np.arange(rank,rank+datasize).astype('uint32')
#b = np.arange(rank+1,rank+1+datasize).astype('uint16')
#c = np.arange(rank+2,rank+2+datasize).astype('float32')
#d = np.arange(rank+3,rank+3+datasize).astype('uint32')
#e = np.arange(rank+4,rank+4+datasize).astype('uint32')

ranksize=5

#d = {n:{'a': a+n, 'b': b, 'c': c, 'd': d+n, 'e': e+n} for n in g.keys()}
d = {n:{'a': a+n, 'b': b+n} for n in range(rank*ranksize,(rank+1)*ranksize)}

append_cell_attributes("data/DGC_forest_attr_test_20200407.h5", "GC", d, io_size=2)
append_cell_attributes("data/DGC_forest_attr_test_20200407.h5", "GC", d, io_size=2)
