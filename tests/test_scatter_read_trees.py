import sys
from neuroh5.io import scatter_read_trees, read_population_ranges, read_cell_attribute_info
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("rank %d / %d" % (rank, size))
sys.stdout.flush()
#(g,n)  = scatter_read_trees("/scratch1/03320/iraikov/dentate/Full_Scale_Control/DG_IN_forest_20190325_compressed.h5", "AAC", io_size=4, comm=comm)
#(g,n)  = scatter_read_trees("/scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DGC_forest_reindex_20190717_compressed.h5", "GC", io_size=8, comm=comm, topology=True)
(g,_) = scatter_read_trees("/scratch1/03320/iraikov/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210203_compressed.h5", 
                           "GC", topology=True, io_size=4, validate=True)

for (gid, tree) in g:
    print ("rank %d: gid %d = %s" % (rank, gid, str(tree)))
