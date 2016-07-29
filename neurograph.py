'''
This script reads a text file with connectivity
and creates a CCS or CRS representation of in an HDF5 file. We assume that
the entries are sorted first by column index and then by row index The
datasets are extensible and GZIP compression was applied.

Supported types of connectivity:

1) Layer-Section-Node connectivity has the following format:

src dest weight layer section node

2) Distance-based connectivity has the following format:

src dest distance

3) Longitudinal+transverse distance-based connectivity has the following format:

src dest longitudinal-distance transverse-distance

'''

import h5py
import numpy as np
import sys, os
import click

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
    print 'dsize = ', dsize
    newshape = (dsize+len(data),)
    print 'newshape = ', newshape
    dset.resize(newshape)
    dset[dsize:] = data
    return dset

@click.group()
def cli():
    return

@cli.command(name="import-lsn")
@click.argument("groupname", type=str, default="lsn")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
def import_lsn(inputfiles, outputfile, groupname, order, colsep):

    col_old = 0
    row_old = 0

    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines()
        f.close()

        if order=='ccs':

            col_ptr    = [0]
            row_idx    = []
            syn_weight = []
            layer      = []
            seg_idx    = []
            node_idx   = []
        
            # read and parse line-by-line

            for l in lines:
                a = l.split(colsep)
                col = int(a[1])-1
                while col_old < col:
                    col_ptr.append(len(syn_weight))
                    col_old = col_old + 1
                row_idx.append(int(a[0])-1)
                syn_weight.append(float(a[2]))
                layer.append(int(a[3]))
                seg_idx.append(int(a[4])-1)
                node_idx.append(int(a[5])-1)

            col_old = col_old + 1

            with h5py.File(outputfile, "a", libver="latest") as h5:

                g1 = h5_get_group (h5, groupname)

                # maxshape=(None,) makes a dataset dimension unlimited.
                # compression=6 applies GZIP compression level 6
                # h5py picks a chunk size for us, but we could also set
                # that manually

                dset = h5_get_dataset(g1, "row_idx", dtype=np.uint32, 
                                      maxshape=(None,), compression=6)
                # if this dataset already contains some data, then
                # calculate the offset and shift the new colptr by
                # that offset.
                row_idx_offset = dset.shape[0]
                dset = h5_concat_dataset(dset, np.asarray(row_idx))

                dset = h5_get_dataset(g1, "col_ptr", dtype=np.uint32, 
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(col_ptr)+row_idx_offset)
                
                # for floating point numbers, it's usally beneficial to apply the
                # bit-shuffle filter before compressing with GZIP
                dset = h5_get_dataset(g1, "Synaptic weight", dtype=np.float32,
                                      maxshape=(None,), compression=6, shuffle=True)
                dset = h5_concat_dataset(dset, np.asarray(syn_weight))
                
                # create an HDF5 enumerated type for the layer information
                mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                           "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
                dt = h5py.special_dtype(enum=(np.uint8, mapping))
                dset = h5_get_dataset(g1, "Layer", dtype=dt, 
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(layer))
                
                dset = h5_get_dataset(g1, "seg_idx", dtype=np.uint16, 
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(seg_idx))
                
                dset = h5_get_dataset(g1, "node_idx", dtype=np.uint16, 
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(node_idx))

        elif order=='crs':

            row_ptr    = [0]
            col_idx    = []
            syn_weight = []
            layer      = []
            seg_idx    = []
            node_idx   = []
                
            for l in lines:
                a = l.split(colsep)
                row = int(a[0])-1
                while row_old < row:
                    row_ptr.append(len(syn_weight))
                    row_old = row_old + 1
                col_idx.append(int(a[1])-1)
                syn_weight.append(float(a[2]))
                layer.append(int(a[3]))
                seg_idx.append(int(a[4])-1)
                node_idx.append(int(a[5])-1)
                
            row_old = row_old + 1

            with h5py.File(outputfile, "a", libver="latest") as h5:

                g1 = h5_get_group (h5, groupname)
                
                dset = h5_get_dataset(g1, "col_idx", dtype=np.uint32,
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(col_idx))
                col_idx_offset = dset.shape[0]

                dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint32,
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(row_ptr)+col_idx_offset)
                    
                dset = h5_get_dataset(g1, "Synaptic weight", dtype=np.float32,
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(syn_weight))
                    
                # create an HDF5 enumerated type for the layer information
                mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                           "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
                dt = h5py.special_dtype(enum=(np.uint8, mapping))
                dset = h5_get_dataset(g1, "Layer", dtype=dt, 
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(layer))
                
                dset = h5_get_dataset(g1, "seg_idx", dtype=np.uint16,
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(seg_idx))
                
                dset = h5_get_dataset(g1, "node_idx", dtype=np.uint16,
                                      maxshape=(None,), compression=6)
                dset = h5_concat_dataset(dset, np.asarray(node_idx))


@cli.command(name="import-dist")
@click.argument("groupname", type=str, default="dist")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
def import_dist(inputfiles, outputfile, groupname, order, colsep):

    col_old = 0
    row_old = 0
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines()
        f.close()

        if order=='ccs':

            col_ptr    = [0]
            row_idx    = []
            dist       = []
        
            # read and parse line-by-line

            for l in lines:
                a = l.split(colsep)
                col = int(a[1])-1
                while col_old < col:
                    col_ptr.append(len(dist))
                    col_old = col_old + 1
                row_idx.append(int(a[0])-1)
                dist.append(float(a[2]))

            col_old = col_old + 1

            with h5py.File(outputfile, "a", libver="latest") as h5:

                g1 = h5_get_group (h5, groupname)

                # maxshape=(None,) makes a dataset dimension unlimited.
                # compression=6 applies GZIP compression level 6
                # h5py picks a chunk size for us, but we could also set
                # that manually

                dset = h5_get_dataset(g1, "row_idx", maxshape=(None,),
                                      dtype=np.uint32, compression=6)
                # if this dataset already contains some data, then
                # calculate the offset and shift the new colptr by
                # that offset.
                row_idx_offset = dset.shape[0]
                dset = h5_concat_dataset(dset, np.asarray(row_idx))

                dset = h5_get_dataset(g1, "col_ptr", maxshape=(None,),
                                      dtype=np.uint32, compression=6)
                dset = h5_concat_dataset(dset, np.asarray(col_ptr)+row_idx_offset)
                
                # for floating point numbers, it's usally beneficial to apply the
                # bit-shuffle filter before compressing with GZIP
                dset = h5_get_dataset(g1, "Distance", 
                                      maxshape=(None,), dtype=np.float32,
                                      compression=6, shuffle=True)
                dset = h5_concat_dataset(dset, np.asarray(dist))
                

        elif order=='crs':

            row_ptr    = [0]
            col_idx    = []
            dist       = []
                
            for l in lines:
                a = l.split(colsep)
                row = int(a[0])-1
                while row_old < row:
                    row_ptr.append(len(dist))
                    row_old = row_old + 1
                col_idx.append(int(a[1])-1)
                dist.append(float(a[2]))
                
            row_old = row_old + 1

            with h5py.File(outputfile, "a", libver="latest") as h5:

                g1 = h5_get_group (h5, groupname)
                
                dset = h5_get_dataset(g1, "col_idx", dtype=np.uint32)
                dset = h5_concat_dataset(dset, np.asarray(col_idx))
                col_idx_offset = dset.shape[0]
                
                dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint32)
                dset = h5_concat_dataset(dset, np.asarray(row_ptr)+col_idx_offset)
                
                dset = h5_get_dataset(g1, "Distance", 
                                      dtype=np.float32)
                dset = h5_concat_dataset(dset, np.asarray(dist))
                    

            

@cli.command(name="import-longtrans")
@click.argument("groupname", type=str, default="longtrans")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
def import_longtrans(inputfiles, outputfile, groupname, order, colsep):

    col_old = 0
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines()
        f.close()

        if order=='ccs':

            col_ptr    = [0]
            row_idx    = []
            longdist   = []
            transdist  = []
        
            # read and parse line-by-line

            for l in lines:
                a = l.split(colsep)
                col = int(a[1])-1
                while col_old < col:
                    col_ptr.append(len(longdist))
                    col_old = col_old + 1
                row_idx.append(int(a[0])-1)
                longdist.append(float(a[2]))
                transdist.append(float(a[3]))

            col_old = col_old + 1

            with h5py.File(outputfile, "a", libver="latest") as h5:

                g1 = h5_get_group (h5, groupname)

                # maxshape=(None,) makes a dataset dimension unlimited.
                # compression=6 applies GZIP compression level 6
                # h5py picks a chunk size for us, but we could also set
                # that manually

                dset = h5_get_dataset(g1, "row_idx", maxshape=(None,),
                                      dtype=np.uint32, compression=6)
                # if this dataset already contains some data, then
                # calculate the offset and shift the new colptr by
                # that offset.
                row_idx_offset = dset.shape[0]
                dset = h5_concat_dataset(dset, np.asarray(row_idx))

                dset = h5_get_dataset(g1, "col_ptr", maxshape=(None,),
                                      dtype=np.uint32, compression=6)
                dset = h5_concat_dataset(dset, np.asarray(col_ptr)+row_idx_offset)
                
                # for floating point numbers, it's usally beneficial to apply the
                # bit-shuffle filter before compressing with GZIP
                dset = h5_get_dataset(g1, "Longitudinal Distance", 
                                      maxshape=(None,), dtype=np.float32,
                                      compression=6, shuffle=True)
                dset = h5_concat_dataset(dset, np.asarray(longdist))

                dset = h5_get_dataset(g1, "Transverse Distance", 
                                      maxshape=(None,), dtype=np.float32,
                                      compression=6, shuffle=True)
                dset = h5_concat_dataset(dset, np.asarray(transdist))
                

        elif order=='crs':

            row_old = 0
            for inputfile in inputfiles:

                row_ptr    = [0]
                col_idx    = []
                longdist   = []
                transdist  = []
                
                for l in lines:
                    a = l.split(colsep)
                    row = int(a[0])-1
                    while row_old < row:
                        row_ptr.append(len(longdist))
                        row_old = row_old + 1
                    col_idx.append(int(a[1])-1)
                    longdist.append(float(a[2]))
                    transdist.append(float(a[3]))
                
                row_old = row_old + 1

                with h5py.File(outputfile, "a", libver="latest") as h5:

                    g1 = h5_get_group (h5, groupname)
                    
                    dset = h5_get_dataset(g1, "col_idx", dtype=np.uint32)
                    dset = h5_concat_dataset(dset, np.asarray(col_idx))
                    col_idx_offset = dset.shape[0]

                    dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint32)
                    dset = h5_concat_dataset(dset, np.asarray(row_ptr)+col_idx_offset)
                    
                    dset = h5_get_dataset(g1, "Longitudinal Distance", 
                                          dtype=np.float32)
                    dset = h5_concat_dataset(dset, np.asarray(longdist))

                    dset = h5_get_dataset(g1, "Transverse Distance", 
                                          dtype=np.float32)
                    dset = h5_concat_dataset(dset, np.asarray(transdist))
                    

            
