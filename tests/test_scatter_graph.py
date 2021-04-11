from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD
#print "rank = ", comm.Get_rank()
#print "size = ", comm.Get_size()

(g, a) = scatter_read_graph("data/dentate_test.h5")
print  (g)

#xprint a
#print g

