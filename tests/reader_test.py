from mpi4py import MPI
from neurotrees.io import read_trees

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()

g = read_trees(MPI._addressof(comm), "data/DGC_forest_test_attrs.h5", "GC")
#pickle.dump( g, open( "DGC_trees.pkl", "wb" ) )
