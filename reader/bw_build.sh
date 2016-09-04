#!/bin/sh
make HDF5_LIBDIR=/opt/cray/hdf5-parallel/1.8.16/GNU/49/lib HDF5_INCDIR=/opt/cray/hdf5-parallel/1.8.16/GNU/49/include \
MPI_INCDIR=$CRAY_MPICH_DIR/include MPI_LIBDIR=$CRAY_MPICH_DIR/lib

