
import h5py
import numpy as np

grp_h5types   = 'H5Types'

def type_path(type_name):
    path = '/%s/%s' % (grp_h5types, type_name)
    return path


def get_type(file_name, type_name):
    dtype = None
    with h5py.File(file_name, "r") as f:
        path = type_path(type_name)
        dtype = f[path]
    return dtype
    

def create_enum_type(file_name, type_name, type_fields, base_type=np.uint8):
    mapping  = { name: idx for name, idx in fields.iteritems() }
    dtype    = h5py.special_dtype(enum=(base_type, mapping))
    path     = type_path(type_name)
    
    with h5py.File(file_name, "a") as f:
        f[path] = dtype
    return dtype
    
    
def create_table_type(f, fields):
    mapping = [ (name, field_dtype) in fields.iteritems() ]
    dtype   = np.dtype(mapping)

    with h5py.File(file_name, "a") as f:
        f[path] = dtype
    return dtype
    
