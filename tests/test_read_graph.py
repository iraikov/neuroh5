import pprint
from mpi4py import MPI
from neuroh5.io import read_graph

g = read_graph("data/dentate_test.h5")
pprint.pprint(list(g[0]['HC']['MC']))

