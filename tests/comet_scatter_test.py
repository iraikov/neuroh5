from mpi4py import MPI
from neurograph.io import scatter_graph
import numpy as np

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

if comm.Get_rank() == 0:
   node_rank_vector = np.loadtxt("parts.4096", dtype=np.uint32)
   node_rank_vector = comm.bcast(node_rank_vector, root=0)
else:
   node_rank_vector = None
   node_rank_vector = comm.bcast(node_rank_vector, root=0)

g = scatter_graph(MPI._addressof(comm), 
                  "/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_Control_MPP.h5", 
                  192, node_rank_vector)

print g.keys()
