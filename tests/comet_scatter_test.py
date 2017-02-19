from mpi4py import MPI
from neurograph.io import scatter_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

g = scatter_graph(MPI._addressof(comm), 
                  "/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_Control_MPP.h5", 
                  192)

print g.keys()
