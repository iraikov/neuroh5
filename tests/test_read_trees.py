import pprint
from mpi4py import MPI
from neuroh5.io import read_trees
#(g,_) = read_trees("data/dentatenet_Network_Clamp_GC_Exc_Sat_DD_S_673410.h5", "GC", topology=True, validate=True)
(g,_) = read_trees("data/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201221_compressed.h5", "GC", topology=True, validate=True)
#(g,_) = read_trees("data/GC_tree_syns_connections_20200214.h5", "GC", topology=False)
#(g,_) = read_trees("data/GC_tree_syns_connections_20181127.h5", "GC", topology=False)
(gid,t) = next(g)
print(gid)
pprint.pprint(t)
#pickle.dump( g, open( "DGC_trees.pkl", "wb" ) )
