from collections import defaultdict
from mpi4py import MPI
from neuroh5.io import scatter_read_graph
from neuroh5.graphlib import load_graph, load_neighbors, neighbor_degrees

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

comm_size = comm.Get_size()


#(graph, n_nodes) = load_graph (comm, "data/dentate_test.h5", 1, map_type=0, node_ranks=None)



(nb_dict, node_index) = load_neighbors (comm, "data/dentate_test.h5", 1)
n_nodes = len(node_index)

node_ranks = defaultdict(list)
for x in node_index:
    rank = x % comm_size
    node_ranks[rank].append(x)

nbdegree_dict = neighbor_degrees (comm, nb_dict, node_ranks)

cc = clustering_coefficient (comm, n_nodes, nb_dict, nbdegree_dict, node_ranks)

#print (graph)

