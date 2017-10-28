from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD

input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170902.h5'
input_file='./data/dentate_test.h5'
input_file='/home/igr/src/model/dentate/datasets/Test_GC_1000/DGC_test_connections_20171019.h5'

#(g,a) = scatter_read_graph(comm, input_file, io_size=1, namespaces=["Attributes"])

(graph, a) = scatter_read_graph(comm,input_file,io_size=1,
                                projections=[('BC', 'MC')],
                                namespaces=['Synapses','Connections'])

#print graph.keys()
edge_dict = {}
edge_iter = graph['MC']['BC']
for (gid,edges) in edge_iter:
    edge_dict[gid] = edges

print edge_dict[1004346]

