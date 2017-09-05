import sys
import functools
from collections import defaultdict
from mpi4py import MPI
from neuroh5.io import scatter_read_graph
from neuroh5.graphlib import make_node_rank_map, read_neighbors, neighbor_degrees, clustering_coefficient
import numpy as np

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

comm_size = comm.Get_size()

input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170728.h5'
input_file='data/dentate_test.h5'

(node_ranks, n_nodes) = make_node_rank_map (comm, input_file, 1)

nb_dict = read_neighbors (comm, input_file, 256, node_ranks)

min_in_degree  = sys.maxint
min_out_degree = sys.maxint
max_in_degree  = 0
max_out_degree = 0

for (k,d) in nb_dict.iteritems():
    print 'gid %d:' % k
    if d.has_key('src'):
        in_degree = len(d['src'])
    else:
        in_degree = 0
    if d.has_key('dst'):
        out_degree = len(d['dst'])
    else:
        out_degree = 0
    min_in_degree  = min(min_in_degree, in_degree)
    min_out_degree = min(min_out_degree, out_degree)
    max_in_degree  = max(max_in_degree, in_degree)
    max_out_degree = max(max_out_degree, out_degree)
    print '    in: %d out: %d' % (in_degree, out_degree)

print 'in degree: min=%d max=%d' % (min_in_degree, max_in_degree)
print 'out degree: min=%d max=%d' % (min_out_degree, max_out_degree)



