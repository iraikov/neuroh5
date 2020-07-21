from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD
rank = comm.rank

input_file='./data/dentate_test.h5'
input_file='/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/DG_Connections_Full_Scale_20180722.h5'
input_file = '/scratch1/03320/iraikov/striped/dentate/Test_GC_1000/DG_Test_GC_1000_connections_20190625_compressed.h5'

#(g,a) = scatter_read_graph(comm, input_file, io_size=1, namespaces=["Attributes"])

(graph, a) = scatter_read_graph(input_file,io_size=8)
                                #projections=[('GC', 'MC'), ('MC', 'MC'), ('AAC', 'MC')],
                                #namespaces=['Synapses','Connections'])
#print graph.keys()
edge_dict = {}
edge_iter = graph['GC']['MC']
for (gid,edges) in edge_iter:
    edge_dict[gid] = edges

print("rank %d: %s" % (rank, str(edge_dict)))

