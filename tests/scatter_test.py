from mpi4py import MPI
from neurograph.io import scatter_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

g = scatter_graph(MPI._addressof(comm), "data/dentate_test.h5", 5, attributes=True)

print g

