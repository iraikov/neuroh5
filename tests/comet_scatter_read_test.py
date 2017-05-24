from mpi4py import MPI
from neurotrees.io import scatter_read_trees, population_ranges
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

path="/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/DGC_forest_syns_20170202.h5"
pr = population_ranges(MPI._addressof(comm), path)
(g,_)  = scatter_read_trees(MPI._addressof(comm), path, "GC", 128,
                            attributes=True, namespace='Synapse_Attributes')
for gid in g.keys():
    val = g[gid]
    dendsyn = np.where(val['Synapse_Attributes.swc_type'] == 5)
    ssec = set(val['Synapse_Attributes.section'][dendsyn])
    msec = set(val['section'])
    if len(ssec) != len(msec):
        print "gid %d: synapse sections length = %d  morphology sections length = %d\n" % (gid, len(ssec), len(msec))
print "rank %d done\n" % rank
MPI.Finalize()
