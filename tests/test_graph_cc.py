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

(node_ranks, n_nodes) = make_node_rank_map (comm, "data/dentate_test.h5", 1)

nb_dict = read_neighbors (comm, "data/dentate_test.h5", 1, node_ranks)

nb_degree_dict = neighbor_degrees (comm, nb_dict, node_ranks)

cc = clustering_coefficient (comm, n_nodes, nb_dict, nb_degree_dict, node_ranks)

print "Clustering coefficient is ", cc


