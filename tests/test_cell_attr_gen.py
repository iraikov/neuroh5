from mpi4py import MPI
from neuroh5.io import read_population_ranges, NeuroH5CellAttrGen

# import mkl
import sys, os, gc
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.rank

input_path = "/scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DGC_forest_syns_20201217_compressed.h5"
synapse_namespace = 'Synapse Attributes'
io_size=20
cache_size=1

if rank == 0:
    print("%d ranks allocated" % comm.size)
    sys.stdout.flush()
it = NeuroH5CellAttrGen(input_path, 'GC', namespace=synapse_namespace, \
                        comm=comm, io_size=io_size, cache_size=cache_size)

for (gid, synapse_dict) in it:
    if gid is not None:
        print('rank %i: gid = %i' % (rank, gid))
        sys.stdout.flush()

