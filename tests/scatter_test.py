from mpi4py import MPI
from neurograph.reader import scatter_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

g = scatter_graph(MPI._addressof(comm), "data/dentate_Full_Scale_Control_test2.h5", 1)

