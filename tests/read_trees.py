from mpi4py import MPI
from neuroh5.io import read_trees

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()

(g,_) = read_trees(MPI._addressof(comm), "data/DGC_forest_test_20170614.h5", "GC")
print g.keys()
#pickle.dump( g, open( "DGC_trees.pkl", "wb" ) )
