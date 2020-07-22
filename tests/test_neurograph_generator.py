import sys,gc,logging
from mpi4py import MPI
from neuroh5.io import NeuroH5ProjectionGen, read_projection_names
import numpy as np
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('neuroh5')

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
path = '/scratch1/03320/iraikov/striped/dentate/Test_GC_1000/DG_Test_GC_1000_connections_20190625_compressed.h5'
path = '/scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_GC_connections_20200703_compressed.h5'


cache_size = 10
destination = 'GC'
sources = ['MC', 'MPP', 'LPP']
#cache_size=cache_size
gg = [ NeuroH5ProjectionGen (path, source, destination, io_size=8, cache_size=5, comm=comm) for source in sources ]
    
for prj_gen_tuple in zip_longest(*gg):
    destination_gid = prj_gen_tuple[0][0]
    if destination_gid is not None:
        logger.info ('rank %d: destination_gid = %d' % (rank, destination_gid))
        for (source, (i,rest)) in zip_longest(sources, prj_gen_tuple):
            logger.info ('rank %d: destination_gid = %d source = %s' % (rank, destination_gid, str(source)))
    else:
        logger.info ('rank %d: destination_gid = %s' % (rank, str(destination_gid)))


