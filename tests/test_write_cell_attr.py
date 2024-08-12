from mpi4py import MPI
import h5py
from neuroh5.io import read_trees, write_cell_attributes, append_cell_attributes, read_cell_attributes
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = np.arange(rank*10,(rank+1)*10).astype('uint32')
b = np.arange(rank*20,(rank+1)*20).astype('float32')

d = {n:{'a': a+n, 'b': b+n} for n in range(rank*5,(rank+1)*5)}

pop_name = 'GC'
output_path = "data/write_cell_attr.h5"

if rank == 0:
    attr_dict = d
else:
    attr_dict = {}

grp_h5types   = 'H5Types'
grp_populations= 'Populations'

path_population_labels = '/%s/Population labels' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types


def h5_get_group (h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset

if rank == 0:
    with h5py.File(output_path, "a") as h5:

        n_pop = 1
        defs = []
        pop_def = (pop_name,0,1000000,0)
        defs.append(pop_def)

        # create an HDF5 enumerated type for the population label
        mapping = { name: idx for name, start, count, idx in defs }
        dt = h5py.special_dtype(enum=(np.uint16, mapping))
        h5[path_population_labels] = dt

        dt = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                       ("Population", h5[path_population_labels].dtype)])
        h5[path_population_range] = dt

        # create an HDF5 compound type for population ranges
        dt = h5[path_population_range].dtype

        g = h5_get_group (h5, grp_h5types)

        dset = h5_get_dataset(g, grp_populations, maxshape=(n_pop,), dtype=dt)
        dset.resize((n_pop,))
        a = np.zeros(n_pop, dtype=dt)
        idx = 0
        for name, start, count, idx in defs:
            a[idx]["Start"] = start
            a[idx]["Count"] = count
            a[idx]["Population"] = idx
            idx += 1

        dset[:] = a
comm.barrier()

write_cell_attributes(output_path, pop_name, attr_dict, namespace='Test Attributes')
print(list(read_cell_attributes(output_path, pop_name, 'Test Attributes')))
