from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD

input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170902.h5'
input_file='./data/dentate_test.h5'
(g,a) = scatter_read_graph(comm, input_file, io_size=2)

print g.keys()
