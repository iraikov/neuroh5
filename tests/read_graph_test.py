from mpi4py import MPI
from neuroh5.io import read_graph, read_graph_serial

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

g = read_graph(comm, "data/dentate_test.h5")
print (g)

g = read_graph_serial("data/dentate_test.h5")
print (g)

#xprint a
#print g

