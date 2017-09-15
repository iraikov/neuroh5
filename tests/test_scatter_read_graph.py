from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD

input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170902.h5'

(g,a) = scatter_read_graph(comm, input_file, io_size=128)

print g.keys()
