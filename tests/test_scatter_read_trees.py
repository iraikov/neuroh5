import sys
from neuroh5.io import scatter_read_trees, read_population_ranges, read_cell_attribute_info
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

path = "/scratch1/03320/iraikov/striped2/MiV/Microcircuit/MiV_Cells_Microcircuit_20220412.h5"

(g,n)  = scatter_read_trees(path, "PYR", io_size=12, comm=comm)

for (gid, tree) in g:
    print (f"rank {rank}: gid {gid} = {np.sum(tree['section'])}")
