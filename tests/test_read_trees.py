from mpi4py import MPI
from neuroh5.io import read_trees

comm = MPI.COMM_WORLD
print("rank = ", comm.Get_rank())

(g,_) = read_trees("data/DGC_forest_test_20170919.h5", "GC")

(gid,t) = next(g)
 
#pickle.dump( g, open( "DGC_trees.pkl", "wb" ) )
