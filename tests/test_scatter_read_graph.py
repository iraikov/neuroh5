from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD

input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/DG_Connections_Full_Scale_20180722.h5'
input_file='./data/dentate_test.h5'

#(g,a) = scatter_read_graph(comm, input_file, io_size=1, namespaces=["Attributes"])

(graph, a) = scatter_read_graph(input_file,io_size=256,
                                projections=[('GC', 'MC'), ('MC', 'MC'), ('AAC', 'MC')],
                                namespaces=['Synapses','Connections'])
print a
print graph.keys()
#print graph.keys()
edge_dict = {}
edge_iter = graph['MC']['AAC']
for (gid,edges) in edge_iter:
    edge_dict[gid] = edges

keys = edge_dict.keys()
if len(keys) > 0:
    print edge_dict[keys[0]]

