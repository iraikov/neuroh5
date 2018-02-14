'''
This script reads a text file in with projection definitions and
creates a representation of it in an HDF5 file.

'''

import click
import h5py
import numpy as np

grp_h5types      = 'H5Types'
grp_projections  = 'Projections'
grp_populations  = 'Populations'
grp_population_projections = 'Population projections'
grp_valid_population_projections = 'Valid population projections'

path_population_projections = '/%s/Population projections' % grp_h5types
path_population_labels = '/%s/Population labels' % grp_h5types



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
@click.argument("projections-file", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
def import_defs(projections_file, outputfile, colsep):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        has_population_projections = False
        if grp_h5types in list(h5.keys()):
            if grp_population_projections in h5[grp_h5types]:
                dt = h5[path_population_projections].dtype
                has_population_projections = True
                
        if not has_population_projections: 
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group (h5, grp_h5types)
        f = open(projections_file)
        lines = f.readlines()
        
        dt = h5[path_population_projections]
        dset = h5_get_dataset(g, grp_valid_population_projections,
                              maxshape=(len(lines),), dtype=dt)
        dset.resize((len(lines),))
        a = np.zeros(len(lines), dtype=dt)
        idx = 0
        for l in lines:
            pre, post, src, dst = l.split(colsep)
            a[idx]["Source"] = int(src)
            a[idx]["Destination"] = int(dst)
            idx += 1

        dset[:] = a

        f.close()
        
