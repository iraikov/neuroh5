// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_index.cc
///
///  Functions for reading and writing cell indices from an HDF5 file.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include <hdf5.h>

#include <cstring>
#include <vector>

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "write_template.hh"

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{

  namespace cell
  {
    
    herr_t read_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     vector<CELL_IDX_T>&  cell_index
     )
    {
      herr_t ierr = 0;
    
      int rank, size;
    
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          assert(file >= 0);

          size_t dset_size = hdf5::dataset_num_elements(comm, file, hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::CELL_INDEX));
          cell_index.resize(dset_size-1);
          ierr = hdf5::read<CELL_IDX_T> (file,
                                         hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::CELL_INDEX),
                                         0, dset_size-1,
                                         CELL_IDX_H5_NATIVE_T,
                                         cell_index, H5P_DEFAULT);
          assert(ierr >= 0);
          ierr = H5Fclose (file);
          assert(ierr == 0);
        }
      
      uint32_t numitems = cell_index.size();
      ierr = MPI_Bcast(&numitems, 1, MPI_UINT32_T, 0, comm);
      assert(ierr == MPI_SUCCESS);
    
      cell_index.resize(numitems);
      ierr = MPI_Bcast(&(cell_index[0]), numitems, MPI_CELL_IDX_T, 0, comm);
      assert(ierr == MPI_SUCCESS);
    
      return ierr;
    }

    
    herr_t append_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const vector<CELL_IDX_T>&  cell_index,
     const hsize_t start
     )
    {
      herr_t ierr = 0;
    
      hid_t file = hdf5::file_open(comm, file_name, rdwr=true);
      assert(file >= 0);

      hsize_t local_index_size = cell_index.size();

      std::vector<uint64_t> index_size_vector;
      index_size_vector.resize(size);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      assert(status == MPI_SUCCESS);

      hsize_t local_index_start = start;
      for (size_t i=0; i<rank; i++)
        {
          local_index_start = local_index_start + index_size_vector[i];
        }
      hsize_t global_index_size = start;
      for (size_t i=0; i<size; i++)
        {
          global_index_size = global_index_size + index_size_vector[i];
        }

      status = hdf5::write<CELL_IDX_T> (file, hdf5::cell_attribute_path(hdf5::POPS, pop_name, hdf5::CELL_INDEX),
                                        global_index_size, local_index_start, local_index_size,
                                        CELL_IDX_H5_NATIVE_T, cell_index, wapl);
      ierr = H5Fclose (file);
      assert(ierr == 0);

      return ierr;
    }

    
  }
}
