#!/bin/sh

LDCXXSHARED="CC -shared" CXX=CC CC=cc CFLAGS="-DUSE_EDGE_DELIM" MPI_LIB=  \
HDF5_LIB=hdf5 HDF5_INCDIR=$HDF5_DIR/include HDF5_LIBDIR=$HDF5_DIR/lib \
pip install -v --upgrade --no-deps --target=/projects/sciteam/baef/site-packages  \
--install-option="--install-scripts=/projects/sciteam/baef/bin"  .
