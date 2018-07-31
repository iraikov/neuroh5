
import h5py
import numpy as np
import neuroh5.types

grp_h5types   = neuroh5.types.grp_h5types


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


def append(file_name, type_name, path, fields):

    dtype = neuroh5.types.get_type(file_name, type_name)
    
    with h5py.File(outputfile, "a", libver="latest") as h5:

        defs = []
        for field_dict in fields:
            elems = field_dict
            ##elem = (name,int(start),int(count),int(idx))
            defs.append(elems)
            
        dset = h5_get_dataset(g, path, maxshape=(inf,), dtype=dtype)
        
        data = np.zeros(len(defs), dtype=dtype)
        
        for i, flds in enumerate(defs):
            data[i][field_name] = flds[field_name]

        h5_concat_dataset(dset, data)
