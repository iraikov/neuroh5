from mpi4py import MPI
from neuroh5.io import scatter_graph

comm = MPI.COMM_WORLD
#print "rank = ", comm.Get_rank()
#print "size = ", comm.Get_size()

(g, a) = scatter_graph(comm, "data/dentate_test.h5", 3)
print (g)

#xprint a
#print g

