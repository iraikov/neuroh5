import pprint
from mpi4py import MPI
from neuroh5.io import read_trees

(g,_) = read_trees("data/dentatenet_Network_Clamp_GC_Exc_Sat_DD_S_673410.h5", "GC", topology=True)

(gid,t) = next(g)
pprint.pprint(t) 
#pickle.dump( g, open( "DGC_trees.pkl", "wb" ) )
