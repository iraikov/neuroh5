'''
This script reads a text file and creates a vanilla CRS representation
in an HDF5 file. "Vanilla" in the sense that the datasets are fixed
size and use contiguous storage layout, and no compression is applied.
'''

row_ptr = [0]
col_idx = []
syn_weight = []
layer = []
seg_idx = []
node_idx = []

f = open('MPPtoDGC.1.crs.csv')

lines = f.readlines()

# read and and parse line-by-line

row_old = 0

for l in lines:
    a = l.split(',')
    row = int(a[0])-1
    if row > row_old:
        row_ptr.append(len(syn_weight))
        row_old = row
    col_idx.append(int(a[1])-1)
    syn_weight.append(float(a[2]))
    layer.append(int(a[3]))
    seg_idx.append(int(a[4])-1)
    node_idx.append(int(a[5])-1)

row_ptr.append(len(syn_weight))

f.close()

import h5py
import numpy as np

with h5py.File("crs.h5","w", libver="latest") as h5:

    dset = h5.create_dataset("row_ptr", (len(row_ptr),), dtype=np.uint32)
    dset[:] = np.asarray(row_ptr)

    dset = h5.create_dataset("col_idx", (len(col_idx),), dtype=np.uint32)
    dset[:] = np.asarray(col_idx)

    dset = h5.create_dataset("Synaptic weight", (len(syn_weight),),
                             dtype=np.float32)
    dset[:] = np.asarray(syn_weight)

    # create an HDF5 enumerated type for the layer information
    mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
               "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
    dt = h5py.special_dtype(enum=(np.uint8, mapping))
    dset = h5.create_dataset("Layer", (len(layer),), dtype=dt)
    dset[:] = np.asarray(layer)

    dset = h5.create_dataset("seg_idx", (len(seg_idx),), dtype=np.uint16)
    dset[:] = np.asarray(seg_idx)

    dset = h5.create_dataset("node_idx", (len(node_idx),), dtype=np.uint16)
    dset[:] = np.asarray(node_idx)

