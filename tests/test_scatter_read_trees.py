from mpi4py import MPI
from neuroh5.io import scatter_read_trees, read_population_ranges, read_cell_attribute_info

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#(g,n)  = scatter_read_trees("/scratch1/03320/iraikov/dentate/Full_Scale_Control/DG_IN_forest_20190325_compressed.h5", "AAC", io_size=4, comm=comm)
(g,n)  = scatter_read_trees("/scratch1/03320/iraikov/dentate/Full_Scale_Control/DGC_forest_reindex_20190717_compressed.h5", "GC", io_size=256, comm=comm, topology=True)

for (gid, tree) in g:
    print ("rank %d: gid %d = %s" % (rank, gid, str(tree)))
