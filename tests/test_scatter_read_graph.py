from mpi4py import MPI
from neuroh5.io import scatter_read_graph

comm = MPI.COMM_WORLD
rank = comm.rank

input_file='./data/dentate_test.h5'

#(g,a) = scatter_read_graph(comm, input_file, io_size=1, namespaces=["Attributes"])

(graph, a) = scatter_read_graph(input_file, io_size=2)
                                #projections=[('GC', 'MC'), ('MC', 'MC'), ('AAC', 'MC')],
                                #namespaces=['Synapses','Connections'])
edge_dict = {}
edge_iter = graph['GC']['MC']
print(f"edge_iter = {edge_iter}")
for (gid,edges) in edge_iter:
    print(f"gid = {gid}")
    edge_dict[gid] = edges

print("rank %d: %s" % (rank, str(edge_dict)))

