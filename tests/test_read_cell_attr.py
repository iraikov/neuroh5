from mpi4py import MPI
from neuroh5.io import read_trees, read_cell_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

va = read_cell_attributes("/home/igr/src/model/dentate/datasets/Full_Scale_Control/dentate_Full_Scale_Control_coords_20170614.h5",
                          "MEC",
                          namespace="Sampled Coordinates")

ks = va.keys()
print ks
if rank == 0:
    print "rank ",rank,": len va.keys = ", len(ks)
    print "rank ",rank,": va[",ks[0]," = ",va[ks[0]].keys()
    for k in va[ks[0]].keys():
        print "rank ",rank,": ",k, " size = ", va[ks[0]][k].size
        print "rank ",rank,": ",k, " = ", va[ks[0]][k]
if rank == 1:
    print "rank ",rank,": len va.keys = ", len(ks)
    print "rank ",rank,": va[",ks[0]," = ",va[ks[0]].keys()
    for k in va[ks[0]].keys():
        print "rank ",rank,": ",k, " size = ", va[ks[0]][k].size
        print "rank ",rank,": ",k, " = ", va[ks[0]][k]



