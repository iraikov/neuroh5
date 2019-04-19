

module load phdf5/1.8.16

HDF5_LIB=hdf5 HDF5_INCDIR=$TACC_HDF5_INC HDF5_LIBDIR=$TACC_HDF5_LIB \
MPI_LIB= HDF5_LIB=hdf5 CC=mpicc CXX=mpicxx \
 pip install --upgrade --user -v --no-deps  .

