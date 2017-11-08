module load python hdf5 mpi4py

MPI_INCDIR=$MPIHOME/include MPI_LIBDIR=$MPIHOME/lib \
HDF5_LIB=hdf5 HDF5_INCDIR=$HDF5HOME/include HDF5_LIBDIR=$HDF5HOME/lib \
 pip install --upgrade --user -v --no-deps .

