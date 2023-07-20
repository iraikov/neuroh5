from mpi4py import MPI
from neuroh5.io import bcast_graph
import numpy as np

comm = MPI.COMM_WORLD

gapjunctions_file_path="/scratch1/03320/iraikov/striped2/dentate/Full_Scale_Control/DG_gapjunctions_20230114.h5"

(graph, a) = bcast_graph(gapjunctions_file_path,
                         namespaces=['Coupling strength', 'Location'],
                         comm=comm)

if comm.rank == 0:
    print(graph['AAC'])

