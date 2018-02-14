from mpi4py import MPI
from neuroh5.io import scatter_read_graph
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



g = scatter_read_graph("/projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_Control_MPP.h5", 
                  128, node_rank_vector)
