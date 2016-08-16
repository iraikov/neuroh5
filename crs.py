'''
This script reads a text file and creates a CRS representation
in an HDF5 file.
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

    dset = h5.create_dataset("row_ptr", (len(row_ptr),), dtype=np.uint64,
                             compression=6)
    dset[:] = np.asarray(row_ptr)

    dset = h5.create_dataset("col_idx", (len(col_idx),), dtype=np.uint32,
                             compression=6)
    dset[:] = np.asarray(col_idx)

    dset = h5.create_dataset("Synaptic weight", (len(syn_weight),),
                             dtype=np.float32, compression=6)
    dset[:] = np.asarray(syn_weight)

    # create an HDF5 enumerated type for the layer information
    mapping = {"GRANULE_LAYER": 1, "INNER_MOLECULAR_LAYER": 2,
               "MIDDLE_MOLECULAR_LAYER": 3, "OUTER_MOLECULAR_LAYER": 4}
    dt = h5py.special_dtype(enum=(np.uint8, mapping))
    h5["/H5Types/Layer tags"] = dt
    dset = h5.create_dataset("Layer", (len(layer),), dtype=dt,
                             compression=6)
    dset[:] = np.asarray(layer)

    dset = h5.create_dataset("seg_idx", (len(seg_idx),), dtype=np.uint16,
                             compression=6)
    dset[:] = np.asarray(seg_idx)

    dset = h5.create_dataset("node_idx", (len(node_idx),), dtype=np.uint16,
                             compression=6)
    dset[:] = np.asarray(node_idx)

    # create an HDF5 enumerated type for the population label
    mapping = { "GC": 0, "MC": 1, "HC": 2, "BC": 3, "AAC": 4,
                "HCC": 5, "NGFC": 6, "MPP": 7, "LPP": 8 }
    dt = h5py.special_dtype(enum=(np.uint16, mapping))
    h5["/H5Types/Population labels"] = dt

    # create an HDF5 compound type for valid combinations of
    # population labels
    dt = np.dtype([("Source", h5["/H5Types/Population labels"].dtype),
                   ("Destination", h5["/H5Types/Population labels"].dtype)])
    h5["/H5Types/Population label combinations"] = dt

    f = open('connectivity.dat')
    lines = f.readlines()

    dset = h5.create_dataset("Valid population combinations", (len(lines),),
                             dtype=dt)
    a = np.zeros(len(lines), dtype=dt)
    idx = 0
    for l in lines:
        src, dst = l.split()
        a[idx]["Source"] = int(src)
        a[idx]["Destination"] = int(dst)
        idx += 1

    dset[:] = a

    f.close()

    # create an HDF5 compound type for population ranges
    dt = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                   ("Population", h5["/H5Types/Population labels"].dtype)])
    h5["/H5Types/Population range"] = dt

    f = open('populations.dat')
    lines = f.readlines()

    dset = h5.create_dataset("Populations", (len(lines),), dtype=dt)
    a = np.zeros(len(lines), dtype=dt)
    idx = 0
    for l in lines:
        start, count, pop = l.split()
        a[idx]["Start"] = int(start)
        a[idx]["Count"] = int(count)
        a[idx]["Population"] = int(pop)
        idx += 1

    dset[:] = a

    f.close()

    # create an 'Node attributes' group
    h5['/Node attributes/Layer'] = h5["Layer"]
    h5['/Node attributes/node_idx'] = h5["node_idx"]
    h5['/Node attributes/Populations'] = h5["Populations"]
    h5['/Node attributes/seg_idx'] = h5["seg_idx"]
    h5['/Node attributes/Synaptic weight'] = h5["Synaptic weight"]

    # create 'Connectivity' group
    h5['/Connectivity/row_ptr'] = h5['row_ptr']
    h5['/Connectivity/col_idx'] = h5['col_idx']
