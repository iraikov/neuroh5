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


def write_final_size_ccs(groupname,outputfile):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g1 = h5_get_group (h5, groupname)
        col_ptr_dset = h5_get_dataset(g1, "col_ptr", dtype=np.uint64, 
                                      maxshape=(None,), compression=6)
        col_ptr_dsize = col_ptr_dset.shape[0]
        row_idx_dset = h5_get_dataset(g1, "row_idx", dtype=np.uint32, 
                                      maxshape=(None,), compression=6)
        row_idx_dsize = row_idx_dset.shape[0]
        newshape = (col_ptr_dsize+1,)
        col_ptr_dset.resize(newshape)
        col_ptr_dset[col_ptr_dsize] = row_idx_dsize
        return col_ptr_dset


def write_final_size_crs(groupname,outputfile):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g1 = h5_get_group (h5, groupname)
        row_ptr_dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint64, 
                                      maxshape=(None,), compression=6)
        row_ptr_dsize = row_ptr_dset.shape[0]
        col_idx_dset = h5_get_dataset(g1, "col_idx", dtype=np.uint32, 
                                      maxshape=(None,), compression=6)
        col_idx_dsize = col_idx_dset.shape[0]
        newshape = (row_ptr_dsize+1,)
        row_ptr_dset.resize(newshape)
        row_ptr_dset[row_ptr_dsize] = col_idx_dsize
        return row_ptr_dset


def import_lsn_lines_ccs (lines,colsep,col_old,groupname,outputfile):

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
        
        dset = h5_get_dataset(g1, "col_ptr", dtype=np.uint64, 
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

    return col_old

def import_lsn_lines_crs (lines,colsep,row_old,groupname,outputfile):

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
        
        dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint64,
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

    return row_old


@cli.command(name="import-lsn")
@click.argument("groupname", type=str, default="lsn")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
@click.option("--bufsize", type=int, default=100000)
def import_lsn(inputfiles, outputfile, groupname, order, colsep, bufsize):

    col_old = 0
    row_old = 0

    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if order=='ccs':
                col_old = import_lsn_lines_ccs(lines, colsep, col_old, groupname, outputfile)
            elif order=='crs':
                row_old = import_lsn_lines_crs(lines, colsep, row_old, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if order=='ccs':
        write_final_size_ccs (groupname, outputfile)
    elif order=='crs':
        write_final_size_crs (groupname, outputfile)



def import_dist_lines_ccs (lines,colsep,col_old,groupname,outputfile):

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
                              dtype=np.uint64, compression=6)
        dset = h5_concat_dataset(dset, np.asarray(col_ptr)+row_idx_offset)
        
        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g1, "Distance", 
                              maxshape=(None,), dtype=np.float32,
                              compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(dist))

    return col_old


def import_dist_lines_crs (lines,colsep,row_old,groupname,outputfile):

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
        
        dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint64)
        dset = h5_concat_dataset(dset, np.asarray(row_ptr)+col_idx_offset)
        
        dset = h5_get_dataset(g1, "Distance", 
                              dtype=np.float32)
        dset = h5_concat_dataset(dset, np.asarray(dist))

    return row_old
            

@cli.command(name="import-dist")
@click.argument("groupname", type=str, default="dist")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
@click.option("--bufsize", type=int, default=100000)
def import_dist(inputfiles, outputfile, groupname, order, colsep, bufsize):

    col_old = 0
    row_old = 0
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if order=='ccs':
                col_old = import_dist_lines_ccs (lines, colsep, col_old, groupname, outputfile)
            elif order=='crs':
                row_old = import_dist_lines_crs (lines, colsep, row_old, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if order=='ccs':
        write_final_size_ccs (groupname, outputfile)
    elif order=='crs':
        write_final_size_crs (groupname, outputfile)



def import_ltdist_lines_ccs (lines,colsep,col_old,groupname,outputfile):

    col_ptr    = [0]
    row_idx    = []
    ldist      = []
    tdist      = []
            
    # read and parse line-by-line

    for l in lines:
        a = l.split(colsep)
        col = int(a[1])-1
        while col_old < col:
            col_ptr.append(len(ldist))
            col_old = col_old + 1
        row_idx.append(int(a[0])-1)
        ldist.append(float(a[2]))
        tdist.append(float(a[3]))

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
                              dtype=np.uint64, compression=6)
        dset = h5_concat_dataset(dset, np.asarray(col_ptr)+row_idx_offset)
        
        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g1, "Longitudinal Distance", 
                              maxshape=(None,), dtype=np.float32,
                              compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(ldist))
        dset = h5_get_dataset(g1, "Transverse Distance", 
                              maxshape=(None,), dtype=np.float32,
                              compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(tdist))

    return col_old

                    
def import_ltdist_lines_crs (lines,colsep,row_old,groupname,outputfile):

    row_ptr    = [0]
    col_idx    = []
    ldist      = []
    tdist      = []
                
    for l in lines:
        a = l.split(colsep)
        row = int(a[0])-1
        while row_old < row:
            row_ptr.append(len(ldist))
            row_old = row_old + 1
        col_idx.append(int(a[1])-1)
        ldist.append(float(a[2]))
        tdist.append(float(a[3]))
                
    row_old = row_old + 1
    
    with h5py.File(outputfile, "a", libver="latest") as h5:
        
        g1 = h5_get_group (h5, groupname)
        
        dset = h5_get_dataset(g1, "col_idx", dtype=np.uint32)
        dset = h5_concat_dataset(dset, np.asarray(col_idx))
        col_idx_offset = dset.shape[0]
        
        dset = h5_get_dataset(g1, "row_ptr", dtype=np.uint64)
        dset = h5_concat_dataset(dset, np.asarray(row_ptr)+col_idx_offset)
        
        dset = h5_get_dataset(g1, "Longitudinal Distance", 
                              dtype=np.float32)
        dset = h5_concat_dataset(dset, np.asarray(ldist))
        dset = h5_get_dataset(g1, "Transverse Distance", 
                              dtype=np.float32)
        dset = h5_concat_dataset(dset, np.asarray(tdist))

    return row_old


@cli.command(name="import-ltdist")
@click.argument("groupname", type=str, default="ltdist")
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--ccs', 'order', flag_value='ccs', default=True)
@click.option('--crs', 'order', flag_value='crs')
@click.option("--colsep", type=str, default=' ')
@click.option("--bufsize", type=int, default=100000)
def import_ltdist(inputfiles, outputfile, groupname, order, colsep, bufsize):

    col_old = 0
    row_old = 0
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if order=='ccs':
                col_old = import_ltdist_lines_ccs (lines, colsep, col_old, groupname, outputfile)
            elif order=='crs':
                row_old = import_ltdist_lines_crs (lines, colsep, row_old, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if order=='ccs':
        write_final_size_ccs (groupname, outputfile)
    elif order=='crs':
        write_final_size_crs (groupname, outputfile)


@cli.command(name="mask-range")
@click.argument("maskname", type=str)
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.argument("arange", type=(int,int))
def mask_range(maskname, inputfile, outputfile, arange):
            
    arange = np.arange(arange[0],arange[1])
    with h5py.File(outputfile, "a", libver="latest") as h5out:
        g2 = h5_get_group (h5out, "Mask")
        dset = h5_get_dataset(g2, maskname, dtype=np.uint32, 
                              maxshape=(None,), compression=6)

        dset.resize(arange.shape)
        dset[:] = arange


@cli.command(name="mask-col")
@click.argument("maskname", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--postrange", type=(int,int), default=None)
def mask_col(maskname, groupname, inputfile, outputfile, postrange):

    with h5py.File(inputfile, "r", libver="latest") as h5:
                
        g1 = h5[groupname]
            
        col_ptr = g1["col_ptr"]
        row_idx = g1["row_idx"]

        col_intervals = []
        col_range     = np.arange(postrange[0],postrange[1])
        for col in col_range:
            col_start = col_ptr[col]
            col_end   = col_ptr[col+1]-1
            col_intervals.append((col_start,col_end))
            
        row_idx_list = []
        for col_int in col_intervals:
            row_idx_list.append(row_idx[col_int[0]:col_int[1]])

        row_idx_data = np.unique(np.concatenate(row_idx_list))
            
    with h5py.File(outputfile, "a", libver="latest") as h5out:
        g2 = h5_get_group (h5out, "Mask")
        dset = h5_get_dataset(g2, maskname, dtype=np.uint32, 
                              maxshape=(None,), compression=6)

        print 'row_idx_data = ', row_idx_data
        print 'row_idx_data.shape = ', row_idx_data.shape
        dset.resize(row_idx_data.shape)
        dset[:] = row_idx_data


@cli.command(name="export-ltdist")
@click.argument("groupname", type=str, default="ltdist")
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--mask-file", type=click.Path(exists=True))
@click.option("--premask", type=str, default=None)
@click.option("--postmask", type=str, default=None)
def export_ltdist(groupname, inputfile, outputfile, mask_file, premask, postmask):

    with h5py.File(inputfile, "r", libver="latest") as h5:
                
        g1 = h5[groupname]
            
        col_ptr   = g1["col_ptr"]
        row_idx   = g1["row_idx"]
        longdist  = g1["Longitudinal Distance"]
        transdist = g1["Transverse Distance"]

        if postmask is not None:
            with h5py.File(mask_file, "r", libver="latest") as h5mask:
                g2 = h5mask["Mask"]
                col_range = (g2[postmask])[:]
        else:
            col_range = np.arange(0,col_ptr.shape[0])

        if premask is not None:
            with h5py.File(mask_file, "r", libver="latest") as h5mask:
                g2 = h5mask["Mask"]
                row_range = (g2[premask])[:]
        else:
            row_range = None

        for col in col_range:

            col_start = col_ptr[col]
            col_end   = col_ptr[col+1]-1
            
            rows       = row_idx[col_start:col_end]
            longdist1  = longdist[col_start:col_end]
            transdist1 = transdist[col_start:col_end]

            if row_range is not None:
                row_inds = np.where(np.in1d(rows, row_range))[0]
            else:
                row_inds = np.arange(0,len(rows))

            f1=open(outputfile, 'a+')
            for row_index in row_inds:
                f1.write ('%d %d %f %f\n' % (col,rows[row_index],longdist[row_index],transdist[row_index]))

        

@cli.command(name="export-mask")
@click.argument("maskfile", type=click.Path(exists=True))
@click.argument("maskname", type=str)
@click.argument("outputfile", type=click.Path())
def export_mask(maskfile, maskname, outputfile):

    with h5py.File(maskfile, "r", libver="latest") as h5mask:
        g1 = h5mask["Mask"]
        maskset = g1[maskname]
        
        f1=open(outputfile, 'a+')
        for index in maskset:
            f1.write ('%d\n' % index)

        
