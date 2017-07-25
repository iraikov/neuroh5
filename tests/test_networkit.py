import itertools
from mpi4py import MPI
from neuroh5.io import read_graph, population_ranges
from networkit import *
from _NetworKit import GraphEvent, GraphUpdater

comm = MPI.COMM_WORLD
print ("rank = ", comm.Get_rank())
print ("size = ", comm.Get_size())

input_file = "data/dentate_test.h5"

nhg = read_graph(comm, input_file)

def prj_stream(nhg):
    for (presyn, prjs) in nhg.items():
        for (postsyn, edges) in prjs.items():
            sources = edges[0]
            destinations = edges[1]
            for (src,dst) in zip(sources,destinations):
                yield (GraphEvent(GraphEvent.EDGE_ADDITION, src, dst, 1.0))

g = Graph(1127650, False, True)
gu = GraphUpdater(g)
gu.update(prj_stream(nhg))
        
overview(g)


