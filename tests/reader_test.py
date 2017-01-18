from mpi4py import MPI
from neurograph.io import read_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()

g = read_graph(MPI._addressof(comm), "data/dentate_test.h5")

