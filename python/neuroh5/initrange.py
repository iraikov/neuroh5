'''
This script reads a text file in with population range definitions and
creates a representation of it in an HDF5 file.

'''

import click
import h5py
import numpy as np

grp_h5types   = 'H5Types'
grp_populations= 'Populations'

path_population_labels = '/%s/Population labels' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types


def h5_get_group (h, groupname):
    if groupname in list(h.keys()):
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in list(g.keys()):
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

@click.group()
def cli():
    return



@cli.command(name="import-defs")
@click.argument("population-file", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
def import_defs(population_file, outputfile, colsep):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        f = open(population_file)
        lines = f.readlines()
        defs = []
        for l in lines:
            name, start, count, idx = l.split(colsep)
            elem = (name,int(start),int(count),int(idx))
            defs.append(elem)
        f.close()
        
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

        dset = h5_get_dataset(g, grp_populations, maxshape=(len(lines),), dtype=dt)
        dset.resize((len(lines),))
        a = np.zeros(len(lines), dtype=dt)
        idx = 0
        for name, start, count, idx in defs:
            a[idx]["Start"] = start
            a[idx]["Count"] = count
            a[idx]["Population"] = idx
            idx += 1

        dset[:] = a


