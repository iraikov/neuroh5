from mpi4py import MPI
from neurograph.io import bcast_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

(g,a) = bcast_graph(MPI._addressof(comm), "data/dentate_test.h5", attributes=True)

print a
print g

