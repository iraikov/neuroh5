import sys, os
from mpi4py import MPI
from neuroh5.io import read_population_ranges, bcast_cell_attributes

# import mkl


comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    print('%i ranks have been allocated' % comm.size)
sys.stdout.flush()

prefix = os.getenv("WORK")
stimulus_path = '%s/Full_Scale_Control/DG_input_spike_trains_20190724_compressed.h5' % prefix
stimulus_namespace = 'Input Spikes A Diag'
sources = ['MPP', 'LPP']
cell_attrs = {}
for source in sources:
    if rank == 0:
        print("reading attributes for %s" % source)
        sys.stdout.flush()
    cell_attr_gen = bcast_cell_attributes(stimulus_path, source, namespace=stimulus_namespace, root=0, comm=comm, mask=set(["Spike Train"]))
    cell_attrs[source] = { gid: attr_dict for gid, attr_dict in cell_attr_gen }
if rank == 0:
    print(cell_attrs)
