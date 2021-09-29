import sys
from neuroh5.io import scatter_read_trees, read_population_ranges, read_cell_attribute_info
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

path = "/scratch1/03320/iraikov/striped2/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_selection_neg2000_neg1500um_phasemod_20210920_compressed.h5"

(g,n)  = scatter_read_trees(path, "GC", io_size=12, comm=comm)

for (gid, tree) in g:
    print (f"rank {rank}: gid {gid} = {np.sum(tree['section'])}")
