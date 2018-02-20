export MPIHOME=/usr/local/Cellar/mpich/3.2_2
export MPI_INCDIR=$MPIHOME/include
export MPI_LIBDIR=$MPIHOME/lib
export HDF5HOME=/usr/local/hdf5
export HDF5_LIB=hdf5
export HDF5_INCDIR=$HDF5HOME/include
export HDF5_LIBDIR=$HDF5HOME/lib

pip install --upgrade -v --no-deps --prefix=$HOME/anaconda2 .
