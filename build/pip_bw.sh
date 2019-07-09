#!/bin/sh

LDCXXSHARED="CC -shared" CXX=CC CC=cc CFLAGS="-DUSE_EDGE_DELIM" MPI_LIB=  \
HDF5_LIB=hdf5 HDF5_INCDIR=$HDF5_DIR/include HDF5_LIBDIR=$HDF5_DIR/lib \
pip2.7 install -v --upgrade --no-deps --target=/projects/sciteam/bayj/site-packages  \
--install-option="--install-scripts=/projects/sciteam/bayj/bin"  .
