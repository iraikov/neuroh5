from mpi4py import MPI
from neuroh5.io import bcast_graph

comm = MPI.COMM_WORLD
print("rank = ", comm.Get_rank())
print("size = ", comm.Get_size())


(g, a) = bcast_graph("/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_Control_gapjunctions.h5", 
                  attributes=True)

if comm.Get_rank() == 0:
    print(a)
    print(list(g.keys()))

