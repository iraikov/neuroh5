import sys,gc
from mpi4py import MPI
from neuroh5.io import NeuroH5ProjectionGen, read_projection_names
import numpy as np
import itertools

def zip_longest(*args, **kwds):
    if hasattr(itertools, 'izip_longest'):
        return itertools.izip_longest(*args, **kwds)
    else:
        return itertools.zip_longest(*args, **kwds)

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print ('%i ranks have been allocated' % comm.size)
sys.stdout.flush()

path = './data/dentate_test_connectivity.h5'
path = '/home/igr/src/model/dentate/datasets/DG_test_connections_20171022.h5'
path = '/home/igr/src/model/dentate/datasets/Test_GC_1000/DGC_test_connections_20171019.h5'
path = '/home/igr/src/model/dentate/datasets/Test_GC_1000/DG_GC_test_connections_20180402.h5'
cache_size = 10
destination = 'GC'
sources = ['MC', 'MPP', 'LPP']
gg = [ NeuroH5ProjectionGen (path, source, destination, cache_size=cache_size, comm=comm) for source in sources ]
    
for prj_gen_tuple in zip_longest(*gg):
    destination_gid = prj_gen_tuple[0][0]
    if destination_gid is not None:
        for (source, (i,rest)) in zip_longest(sources, prj_gen_tuple):
            print ('source = ', source)
            print ('i = ', i)
            print ('rest = ', rest)

if rank == 0:
    import h5py
    count = 0
    f = h5py.File(path, 'r+')
    if 'test' in f:
        count = f['test'][()]
        del(f['test'])
    f['test'] = count+1
comm.barrier()


#g = NeuroH5ProjectionGen (comm, 'data/dentate_test.h5', 'BC', 'MC')
#for (i,j,attr) in g:
#        if i is not None:
#            print i, j


