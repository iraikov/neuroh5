'''
This script reads a text file with connectivity and creates a
Destination Block Storage representation of in an HDF5 file. We assume
that the entries are sorted first by column index and then by row
index The datasets are extensible and GZIP compression was applied.

Supported types of connectivity:

1) Layer-Section-Node connectivity has the following format:

src dest weight layer section node

2) Distance-based connectivity has the following format:

src dest distance

3) Longitudinal+transverse distance-based connectivity has the following format:

src dest longitudinal-distance transverse-distance

4) Gap junction distance-based connectivity has the following format:

src dest src-branch src-sec dest-branch dest-sec weight

'''

import h5py
import numpy as np
import sys, os
import click

grp_h5types      = 'H5Types'
grp_projections  = 'Projections'
grp_connectivity = 'Connectivity'
grp_populations  = 'Populations'
grp_attributes   = 'Attributes'
grp_node   = 'Node'
grp_edge   = 'Edge'
grp_population_projections = 'Population projections'

path_layer_tags = '/%s/Layer tags' % grp_h5types
path_population_labels = '/%s/Population labels' % grp_h5types
path_population_projections = '/%s/Population projections' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types

path_node_attrs   = '%s/Node' % grp_attributes
path_edge_attrs   = '%s/Edge' % grp_attributes

src_idx     = 'Source Index'
dst_idx     = 'Destination Block Index'
dst_blk_ptr = 'Destination Block Pointer'
dst_ptr     = 'Destination Pointer'

attr_layer        = 'Layer'
attr_syn_weight   = 'Synaptic Weight'
attr_seg_idx      = 'Segment Index'
attr_seg_pt_idx   = 'Segment Point Index'
attr_long_dist    = 'Longitudinal Distance'
attr_trans_dist   = 'Transverse Distance'
attr_dist         = 'Distance'

attr_gj_weight     = 'Weight'
attr_gj_src_branch = 'Source Branch'
attr_gj_dst_branch = 'Destination Branch'
attr_gj_src_sec    = 'Source Section'
attr_gj_dst_sec    = 'Destination Section'


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


def write_population_ids(source,dest,groupname,outputfile):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        dset = h5_get_dataset(g1, 'Source Population', dtype=np.uint32, maxshape=(1,))
        dset.resize((1,))
        dset[0] = source
        dset = h5_get_dataset(g1, 'Destination Population', dtype=np.uint32, maxshape=(1,))
        dset.resize((1,))
        dset[0] = dest


def write_final_size_dbs(groupname,outputfile):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        g2 = h5_get_group (g1, grp_connectivity)

        src_idx_dset = h5_get_dataset(g2, src_idx)
        src_idx_dsize = src_idx_dset.shape[0]

        dst_ptr_dset = h5_get_dataset(g2, dst_ptr)
        dst_ptr_dsize = dst_ptr_dset.shape[0]
        newshape = (dst_ptr_dsize+1,)
        dst_ptr_dset.resize(newshape)
        dst_ptr_dset[dst_ptr_dsize] = src_idx_dsize

        dst_blk_ptr_dset = h5_get_dataset(g2, dst_blk_ptr)
        dst_blk_ptr_dsize = dst_blk_ptr_dset.shape[0]
        newshape = (dst_blk_ptr_dsize+1,)
        dst_blk_ptr_dset.resize(newshape)
        dst_blk_ptr_dset[dst_blk_ptr_dsize] = dst_ptr_dsize+1

        return dst_ptr_dset


def write_connectivity_dbs (g, l_src_idx, l_dst_ptr, dst_min):
    
    g1 = h5_get_group(g, grp_connectivity)

    # maxshape=(None,) makes a dataset dimension unlimited.
    # compression=6 applies GZIP compression level 6
    # h5py picks a chunk size for us, but we could also set
    # that manually
    dset = h5_get_dataset(g1, src_idx, dtype=np.uint32, 
                          maxshape=(None,), compression=6)
    # if this dataset already contains some data, then
    # calculate the offset and shift the new dst_ptr by
    # that offset
    src_idx_offset = dset.shape[0]
    dset = h5_concat_dataset(dset, np.asarray(l_src_idx))
    
    dset = h5_get_dataset(g1, dst_ptr, dtype=np.uint64, 
                          maxshape=(None,), compression=6)
    dst_ptr_offset = dset.shape[0]
    dset = h5_concat_dataset(dset, np.asarray(l_dst_ptr)+src_idx_offset)
    
    dset = h5_get_dataset(g1, dst_blk_ptr, dtype=np.uint32, 
                          maxshape=(None,), compression=6)
    dset = h5_concat_dataset(dset, np.asarray([dst_ptr_offset]))
    
    dset = h5_get_dataset(g1, dst_idx, dtype=np.uint32, 
                          maxshape=(None,), compression=6)
    dset = h5_concat_dataset(dset, np.asarray([dst_min]))


def import_lsn_lines_dbs (lines,source_base,dest_base,colsep,offset,groupname,outputfile):
    
    l_dst_ptr    = [0]
    l_src_idx    = []
    l_dist       = []
    l_layer      = []
    l_seg_idx    = []
    l_seg_pt_idx = []
    l_syn_weight = []
        
    # read and parse line-by-line

    dst_min = -1
    dst_old = -1
    for l in lines:
        a = l.split(colsep)
        src = int(a[0])+offset-source_base
        dst = int(a[1])+offset-dest_base
        if dst_min < 0:
            dst_min = dst
        else:
            dst_min = min(dst_min,dst)
        if dst_old < 0: 
            dst_old = dst
        else:
            while dst_old < dst:
                dst_old = dst_old + 1
                l_dst_ptr.append(len(l_src_idx))
        l_src_idx.append(src)
        l_dist.append(float(a[2]))
        l_layer.append(int(a[3]))
        l_seg_idx.append(int(a[4])-1)
        l_seg_pt_idx.append(int(a[5])-1)
        l_syn_weight.append(float(a[6]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        
        write_connectivity_dbs (g1, l_src_idx, l_dst_ptr, dst_min)
        
        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        g2 = h5_get_group(g1, grp_attributes)
        g3 = h5_get_group(g2, grp_edge)

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g3, attr_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_dist))

        dset = h5_get_dataset(g3, attr_layer, dtype=dt, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_layer))
        
        dset = h5_get_dataset(g3, attr_seg_idx, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_seg_idx))
        
        dset = h5_get_dataset(g3, attr_seg_pt_idx, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_seg_pt_idx))

        dset = h5_get_dataset(g3, attr_syn_weight, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_syn_weight))




@cli.command(name="import-lsn")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--dbs', 'layout', flag_value='dbs', default=True)
@click.option('--sbs', 'layout', flag_value='sbs')
@click.option('--relative-source', 'indextype_src', flag_value='rel', default=True)
@click.option('--absolute-source', 'indextype_src', flag_value='abs')
@click.option('--relative-dest', 'indextype_dst', flag_value='rel', default=True)
@click.option('--absolute-dest', 'indextype_dst', flag_value='abs')
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_lsn(inputfiles, outputfile, source, dest, groupname, layout, indextype_src, indextype_dst, colsep, offset, bufsize):

    population_mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                           "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt
            
            # create an HDF5 enumerated type for the population label
            dt = h5py.special_dtype(enum=(np.uint16, population_mapping))
            h5[path_population_labels] = dt

            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group(h5, grp_h5types)
        dset = h5_get_dataset(g, grp_populations)
        population_defns = {}
        for p in dset[:]:
            population_defns[p[2]] = (p[0], p[1])

    src_index = population_mapping[source]
    dst_index = population_mapping[dest]

    if indextype_src == 'rel':
        src_base = 0
    else:
        src_base = int((population_defns[src_index])[0])
    if indextype_dst == 'rel':
        dst_base = 0
    else:
        dst_base = int((population_defns[dst_index])[0])

    write_population_ids (src_index, dst_index, groupname, outputfile)
        
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if layout=='dbs':
                import_lsn_lines_dbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            elif layout=='sbs':
                import_lsn_lines_sbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if layout=='dbs':
        write_final_size_dbs (groupname, outputfile)
    elif layout=='sbs':
        write_final_size_sbs (groupname, outputfile)



def import_ltdist_lines_dbs (lines,source_base,dest_base,colsep,offset,groupname,outputfile):

    l_dst_ptr    = [0]
    l_src_idx    = []
    l_ldist      = []
    l_tdist      = []
        
    # read and parse line-by-line

    dst_min = -1
    dst_old = -1
    for l in lines:
        a = l.split(colsep)
        src = int(float(a[0]))+offset-source_base
        dst = int(float(a[1]))+offset-dest_base
        print 'src = ', src
        if dst_min < 0:
            dst_min = dst
        else:
            dst_min = min(dst_min,dst)
        if dst_old < 0: 
            dst_old = dst
        else:
            while dst_old < dst:
                dst_old = dst_old + 1
                l_dst_ptr.append(len(l_src_idx))
        l_src_idx.append(src)
        l_ldist.append(float(a[2]))
        l_tdist.append(float(a[3]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        
        write_connectivity_dbs (g1, l_src_idx, l_dst_ptr, dst_min)
        
        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        g2 = h5_get_group(g1, grp_attributes)
        g3 = h5_get_group(g2, grp_edge)


        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g3, attr_long_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_ldist))

        dset = h5_get_dataset(g3, attr_trans_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_tdist))


@cli.command(name="import-ltdist")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--dbs', 'layout', flag_value='dbs', default=True)
@click.option('--sbs', 'layout', flag_value='sbs')
@click.option('--relative-source', 'indextype_src', flag_value='rel', default=True)
@click.option('--absolute-source', 'indextype_src', flag_value='abs')
@click.option('--relative-dest', 'indextype_dst', flag_value='rel', default=True)
@click.option('--absolute-dest', 'indextype_dst', flag_value='abs')
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_ltdist(inputfiles, outputfile, source, dest, groupname, layout, indextype_src, indextype_dst, colsep, offset, bufsize):

    print "offset = ", offset
    population_mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                           "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt
            
            # create an HDF5 enumerated type for the population label
            dt = h5py.special_dtype(enum=(np.uint16, population_mapping))
            h5[path_population_labels] = dt
            
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group(h5, grp_h5types)
        dset = h5_get_dataset(g, grp_populations)
        population_defns = {}
        for p in dset[:]:
            population_defns[p[2]] = (p[0], p[1])

    src_index = population_mapping[source]
    dst_index = population_mapping[dest]

    if indextype_src == 'rel':
        src_base = 0
    else:
        src_base = int((population_defns[src_index])[0])
    if indextype_dst == 'rel':
        dst_base = 0
    else:
        dst_base = int((population_defns[dst_index])[0])

    write_population_ids (src_index, dst_index, groupname, outputfile)
        
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if layout=='dbs':
                import_ltdist_lines_dbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            elif layout=='sbs':
                import_ltdist_lines_sbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if layout=='dbs':
        write_final_size_dbs (groupname, outputfile)
    elif layout=='sbs':
        write_final_size_sbs (groupname, outputfile)


def import_dist_lines_dbs (lines,source_base,dest_base,colsep,offset,groupname,outputfile):

    l_dst_ptr    = [0]
    l_src_idx    = []
    l_dist       = []
        
    # read and parse line-by-line

    dst_min = -1
    dst_old = -1
    for l in lines:
        a = l.split(colsep)
        src = int(a[0])+offset-source_base
        dst = int(a[1])+offset-dest_base
        if dst_min < 0:
            dst_min = dst
        else:
            dst_min = min(dst_min,dst)
        if dst_old < 0: 
            dst_old = dst
        else:
            while dst_old < dst:
                dst_old = dst_old + 1
                l_dst_ptr.append(len(l_src_idx))
        l_src_idx.append(src)
        l_dist.append(float(a[2]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        
        write_connectivity_dbs (g1, l_src_idx, l_dst_ptr, dst_min)
        
        # create an HDF5 enumerated type for the layer information
        mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                   "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
        dt = h5py.special_dtype(enum=(np.uint8, mapping))

        g2 = h5_get_group(g1, grp_attributes)
        g3 = h5_get_group(g2, grp_edge)

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g3, attr_dist, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_dist))


@cli.command(name="import-dist")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--dbs', 'layout', flag_value='dbs', default=True)
@click.option('--sbs', 'layout', flag_value='sbs')
@click.option('--relative-source', 'indextype_src', flag_value='rel', default=True)
@click.option('--absolute-source', 'indextype_src', flag_value='abs')
@click.option('--relative-dest', 'indextype_dst', flag_value='rel', default=True)
@click.option('--absolute-dest', 'indextype_dst', flag_value='abs')
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_dist(inputfiles, outputfile, source, dest, groupname, layout, indextype_src, indextype_dst, colsep, offset, bufsize):

    population_mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                           "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt
            
            # create an HDF5 enumerated type for the population label
            dt = h5py.special_dtype(enum=(np.uint16, population_mapping))
            h5[path_population_labels] = dt
            
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group(h5, grp_h5types)
        dset = h5_get_dataset(g, grp_populations)
        population_defns = {}
        for p in dset[:]:
            population_defns[p[2]] = (p[0], p[1])

    src_index = population_mapping[source]
    dst_index = population_mapping[dest]

    if indextype_src == 'rel':
        src_base = 0
    else:
        src_base = int((population_defns[src_index])[0])
    if indextype_dst == 'rel':
        dst_base = 0
    else:
        dst_base = int((population_defns[dst_index])[0])

    write_population_ids (src_index, dst_index, groupname, outputfile)
        
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if layout=='dbs':
                import_dist_lines_dbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            elif layout=='sbs':
                import_dist_lines_sbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if layout=='dbs':
        write_final_size_dbs (groupname, outputfile)
    elif layout=='sbs':
        write_final_size_sbs (groupname, outputfile)

        
## src dest src-branch src-sec dest-branch dest-sec weight
def import_gj_lines_dbs (lines,source_base,dest_base,colsep,offset,groupname,outputfile):
    
    l_dst_ptr    = [0]
    l_src_idx    = []
    l_src_branch = []
    l_src_sec    = []
    l_dst_branch = []
    l_dst_sec    = []
    l_weight     = []
        
    # read and parse line-by-line

    dst_min = -1
    dst_old = -1
    for l in lines:
        a = l.split(colsep)
        if a[0] == '':
            del a[0]
        src = int(float(a[0]))+offset-source_base
        dst = int(float(a[1]))+offset-dest_base
        if dst_min < 0:
            dst_min = dst
        else:
            dst_min = min(dst_min,dst)
        if dst_old < 0: 
            dst_old = dst
        else:
            while dst_old < dst:
                dst_old = dst_old + 1
                l_dst_ptr.append(len(l_src_idx))
        l_src_idx.append(src)
        l_src_branch.append(int(float(a[2])))
        l_src_sec.append(int(float(a[3])))
        l_dst_branch.append(int(float(a[4])))
        l_dst_sec.append(int(float(a[5])))
        l_weight.append(float(a[6]))
        

    with h5py.File(outputfile, "a", libver="latest") as h5:

        g = h5_get_group (h5, grp_projections)
        g1 = h5_get_group (g, groupname)
        
        write_connectivity_dbs (g1, l_src_idx, l_dst_ptr, dst_min)

        g2 = h5_get_group(g1, grp_attributes)
        g3 = h5_get_group(g2, grp_edge)

        # for floating point numbers, it's usally beneficial to apply the
        # bit-shuffle filter before compressing with GZIP
        dset = h5_get_dataset(g3, attr_gj_weight, dtype=np.float32,
                              maxshape=(None,), compression=6, shuffle=True)
        dset = h5_concat_dataset(dset, np.asarray(l_weight))
        
        dset = h5_get_dataset(g3, attr_gj_src_branch, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_src_branch))

        dset = h5_get_dataset(g3, attr_gj_src_sec, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_src_sec))

        dset = h5_get_dataset(g3, attr_gj_dst_branch, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_dst_branch))

        dset = h5_get_dataset(g3, attr_gj_dst_sec, dtype=np.uint16, 
                              maxshape=(None,), compression=6)
        dset = h5_concat_dataset(dset, np.asarray(l_dst_sec))

        
@cli.command(name="import-gapjunction")
@click.argument("source", type=str)
@click.argument("dest", type=str)
@click.argument("groupname", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option('--dbs', 'layout', flag_value='dbs', default=True)
@click.option('--sbs', 'layout', flag_value='sbs')
@click.option('--relative-source', 'indextype_src', flag_value='rel', default=True)
@click.option('--absolute-source', 'indextype_src', flag_value='abs')
@click.option('--relative-dest', 'indextype_dst', flag_value='rel', default=True)
@click.option('--absolute-dest', 'indextype_dst', flag_value='abs')
@click.option("--colsep", type=str, default=None)
@click.option("--offset", type=int, default=0)
@click.option("--bufsize", type=int, default=100000)
def import_gapjunction(inputfiles, outputfile, source, dest, groupname, layout, indextype_src, indextype_dst, colsep, offset, bufsize):

    population_mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                           "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
    layer_mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                     "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}

    with h5py.File(outputfile, "a", libver="latest") as h5:
        if not (grp_h5types in h5.keys()):
            # create an HDF5 enumerated type for the layer information
            dt = h5py.special_dtype(enum=(np.uint8, layer_mapping))
            h5[path_layer_tags] = dt
            
            # create an HDF5 enumerated type for the population label
            dt = h5py.special_dtype(enum=(np.uint16, population_mapping))
            h5[path_population_labels] = dt
            
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group(h5, grp_h5types)
        dset = h5_get_dataset(g, grp_populations)
        population_defns = {}
        for p in dset[:]:
            population_defns[p[2]] = (p[0], p[1])

    src_index = population_mapping[source]
    dst_index = population_mapping[dest]

    if indextype_src == 'rel':
        src_base = 0
    else:
        src_base = int((population_defns[src_index])[0])
    if indextype_dst == 'rel':
        dst_base = 0
    else:
        dst_base = int((population_defns[dst_index])[0])

    write_population_ids (src_index, dst_index, groupname, outputfile)
        
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)

        while lines:
            if layout=='dbs':
                import_gj_lines_dbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            elif layout=='sbs':
                import_gj_lines_sbs(lines, src_base, dst_base, colsep, offset, groupname, outputfile)
            lines = f.readlines(bufsize)

        f.close()

    if layout=='dbs':
        write_final_size_dbs (groupname, outputfile)
    elif layout=='sbs':
        write_final_size_sbs (groupname, outputfile)




@cli.command(name="import-globals")
@click.argument("population-file", type=click.Path(exists=True))
@click.argument("connectivity-file", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
def import_globals(population_file, connectivity_file, outputfile, colsep):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        if grp_h5types in h5.keys():
            dt = h5[path_population_projections].dtype
        else: 
            # create an HDF5 enumerated type for the layer information
            mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
                       "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
            dt = h5py.special_dtype(enum=(np.uint8, mapping))
            h5[path_layer_tags] = dt

            # create an HDF5 enumerated type for the population label
            mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                        "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
            dt = h5py.special_dtype(enum=(np.uint16, mapping))
            h5[path_population_labels] = dt
        
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

            dt = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                           ("Population", h5[path_population_labels].dtype)])
            h5[path_population_range] = dt

        g = h5_get_group (h5, grp_h5types)
        f = open(connectivity_file)
        lines = f.readlines()
        
        dt = h5[path_population_projections]
        dset = h5_get_dataset(g, "Valid population projections", 
                              maxshape=(len(lines),), dtype=dt)
        dset.resize((len(lines),))
        a = np.zeros(len(lines), dtype=dt)
        idx = 0
        for l in lines:
            label, src, dst = l.split(colsep)
            a[idx]["Source"] = int(src)
            a[idx]["Destination"] = int(dst)
            idx += 1

        dset[:] = a

        f.close()
        
        # create an HDF5 compound type for population ranges
        dt = h5[path_population_range].dtype
        
        f = open(population_file)
        lines = f.readlines()
        
        dset = h5_get_dataset(g, grp_populations, maxshape=(len(lines),), dtype=dt)
        dset.resize((len(lines),))
        a = np.zeros(len(lines), dtype=dt)
        idx = 0
        for l in lines:
            start, count, pop = l.split(colsep)
            a[idx]["Start"] = int(start)
            a[idx]["Count"] = int(count)
            a[idx]["Population"] = int(pop)
            idx += 1

        dset[:] = a

        f.close()

@cli.command(name="import-connectivity")
@click.argument("connectivity-file", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
@click.option("--colsep", type=str, default=' ')
def import_connectivity(connectivity_file, outputfile, colsep):

    with h5py.File(outputfile, "a", libver="latest") as h5:

        if (grp_h5types in h5.keys()) & (grp_population_projections in h5[grp_h5types].keys()):
            dt = h5[path_population_projections].dtype
        else: 
            # create an HDF5 compound type for valid combinations of
            # population labels
            dt = np.dtype([("Source", h5[path_population_labels].dtype),
                           ("Destination", h5[path_population_labels].dtype)])
            h5[path_population_projections] = dt

        g = h5_get_group (h5, grp_h5types)
        f = open(connectivity_file)
        lines = f.readlines()
        
        dt = h5[path_population_projections]
        dset = h5_get_dataset(g, "Valid population projections", 
                              maxshape=(len(lines),), dtype=dt)
        dset.resize((len(lines),))
        a = np.zeros(len(lines), dtype=dt)
        idx = 0
        for l in lines:
            label, src, dst = l.split(colsep)
            a[idx]["Source"] = int(src)
            a[idx]["Destination"] = int(dst)
            idx += 1

        dset[:] = a

        f.close()
        
