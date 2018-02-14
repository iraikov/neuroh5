import matplotlib
matplotlib.use('Agg')
from neuroh5.io import read_graph_serial
from networkit import *

input_file = "data/dentate_test.h5"
input_file = "/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170728.h5"

print(("maximum number of threads is %d\n" % engineering.getMaxNumberOfThreads()))
sys.stdout.flush()

def prj_stream(nhg):
    for (presyn, prjs) in list(nhg.items()):
        for (postsyn, edges) in list(prjs.items()):
            sources = edges[0]
            destinations = edges[1]
            for (src,dst) in zip(sources,destinations):
                yield (src, dst)

setNumberOfThreads(16)

nhg = read_graph_serial(input_file)

g = Graph(1127650, False, True)
count=0
for (i,j) in prj_stream(nhg):
    g.addEdge(i,j)
    count += 1

overview(g)
sys.stdout.flush()

pf = profiling.Profile.create(g, preset="minimal")
pf.output("HTML",".")
