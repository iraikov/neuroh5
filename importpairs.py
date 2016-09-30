'''
This script reads a text file with connectivity and creates a
source-destination pair representation of in an HDF5 file. We assume
that the entries are sorted first by column index and then by row
index The datasets are extensible and GZIP compression was applied.

'''

import h5py
import numpy as np
import sys, os
import click

dset_src          = 'Source'
dset_dst          = 'Destination'
attr_layer        = 'Layer'
attr_syn_weight   = 'Synaptic Weight'
attr_seg_idx      = 'Segment Index'
attr_seg_pt_idx   = 'Segment Point Index'
attr_dist         = 'Distance'

grp_projections  = 'Projections'
grp_connectivity = 'Connectivity'
grp_h5types      = 'H5Types'

path_layer_tags = '/%s/Layer tags' % grp_h5types

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

@click.group()
def cli():
    return



def write_connectivity_pairs (g, l_src, l_dst):

    # maxshape=(None,) makes a dataset dimension unlimited.
    # compression=6 applies GZIP compression level 6
    # h5py picks a chunk size for us, but we could also set
    # that manually
    dset = h5_get_dataset(g, dset_src, dtype=np.uint32, 
                          maxshape=(None,), compression=6)
    # if this dataset already contains some data, then
    # calculate the offset and shift the new dst_ptr by
    # that offset
    dset = h5_concat_dataset(dset, np.asarray(l_src))
    
    dset = h5_get_dataset(g, dset_dst, dtype=np.uint32, 
                              maxshape=(None,), compression=6)
    dset = h5_concat_dataset(dset, np.asarray(l_dst))


def import_lsn_lines_pairs (lines,colsep,groupname,outputfile):
    
    l_src       = []
    l_dst       = []
    l_dist       = []
    l_layer      = []
    l_seg_idx    = []
    l_seg_pt_idx = []
    l_syn_weight = []
        
    # read and parse line-by-line

    for l in lines:
        a = l.split(colsep)
        src = int(a[0])-1
        dst = int(a[1])-1
        l_src.append(src)
        l_dst.append(dst)
        l_dist.append(float(a[2]))
        l_layer.append(int(a[3]))
        l_seg_idx.append(int(a[4])-1)
        l_seg_pt_idx.append(int(a[5])-1)
        l_syn_weight.append(float(a[6]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)

        write_connectivity_pairs (g1, l_src, l_dst)

        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g1, attr_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_dist))

        dset = h5_get_dataset(g1, attr_layer, dtype=dt, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_layer))
        
        dset = h5_get_dataset(g1, attr_seg_idx, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_seg_idx))
        
        dset = h5_get_dataset(g1, attr_seg_pt_idx, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_seg_pt_idx))

        dset = h5_get_dataset(g1, attr_syn_weight, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_syn_weight))




@cli.command(name="import-lsn")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_lsn(inputfiles, outputfile, source, dest, groupname, colsep, offset, bufsize):

    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt

    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            import_lsn_lines_pairs(lines, colsep, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()


def import_ltdist_lines_pairs (lines,colsep,groupname,outputfile):
    
    l_src       = []
    l_dst       = []
    l_ldist     = []
    l_tdist     = []
        
    # read and parse line-by-line

    for l in lines:
        a = l.split(colsep)
        src = int(a[0])-1
        dst = int(a[1])-1
        l_src.append(src)
        l_dst.append(dst)
        l_ldist.append(float(a[2]))
        l_tdist.append(float(a[3]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)

        write_connectivity_pairs (g1, l_src, l_dst)

        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g1, attr_ldist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_ldist))

        dset = h5_get_dataset(g1, attr_tdist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_tdist))




@cli.command(name="import-ltdist")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_ltdist(inputfiles, outputfile, source, dest, groupname, colsep, offset, bufsize):

    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt

    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            import_ltdist_lines_pairs(lines, colsep, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()


def import_dist_lines_pairs (lines,colsep,groupname,outputfile):
    
    l_src       = []
    l_dst       = []
    l_dist     = []
        
    # read and parse line-by-line

    for l in lines:
        a = l.split(colsep)
        src = int(a[0])-1
        dst = int(a[1])-1
        l_src.append(src)
        l_dst.append(dst)
        l_dist.append(float(a[2]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)

        write_connectivity_pairs (g1, l_src, l_dst)

        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g1, attr_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_dist))




@cli.command(name="import-dist")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_dist(inputfiles, outputfile, source, dest, groupname, colsep, offset, bufsize):

    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt

    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            import_dist_lines_pairs(lines, colsep, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

