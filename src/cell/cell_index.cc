// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_index.cc
///
///  Functions for reading and writing cell indices from an HDF5 file.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include <hdf5.h>

#include <cstring>
#include <vector>

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "exists_dataset.hh"
#include "file_access.hh"
#include "create_group.hh"
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
    
    herr_t create_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     const size_t         chunk_size = 1000
     )
    {
      herr_t ierr = 0;
      int srank, ssize; size_t rank, size;
    
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      assert(srank >= 0);
      assert(ssize > 0);
      
      rank = (size_t)srank;
      size = (size_t)ssize;
      
      string attr_prefix = hdf5::cell_attribute_prefix(attr_name_space, pop_name);
      string attr_path   = hdf5::cell_attribute_path(attr_name_space, pop_name, hdf5::CELL_INDEX);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
          assert(file >= 0);

          bool has_group=false, has_index=false;

          has_group = hdf5::exists_dataset (file, attr_prefix) > 0;
          if (!has_group)
            {
              ierr = hdf5::create_group (file, attr_prefix);
              assert(ierr == 0);
            }
          else
            {
              has_index = hdf5::exists_dataset (file, attr_path) > 0;
            }

          if (!has_index)
            {
              
              hsize_t maxdims[1] = {H5S_UNLIMITED};
              hsize_t cdims[1]   = {chunk_size}; /* chunking dimensions */		
              hsize_t initial_size = 0;
              
              hid_t plist  = H5Pcreate (H5P_DATASET_CREATE);
              ierr = H5Pset_chunk(plist, 1, cdims);
              assert(ierr == 0);

              ierr = H5Pset_alloc_time(plist, H5D_ALLOC_TIME_EARLY);
              assert(ierr == 0);
              
              hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
              assert(lcpl >= 0);
              assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
              
              hid_t mspace = H5Screate_simple(1, &initial_size, maxdims);
              assert(mspace >= 0);
              hid_t dset = H5Dcreate2(file, attr_path.c_str(), CELL_IDX_H5_FILE_T,
                                      mspace, lcpl, plist, H5P_DEFAULT);
              assert(H5Dclose(dset) >= 0);
              assert(H5Sclose(mspace) >= 0);
              
              ierr = H5Fclose (file);
              assert(ierr == 0);
            }
        }

      return ierr;
    }
    
    herr_t read_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     vector<CELL_IDX_T>&  cell_index
     )
    {
      herr_t ierr = 0;
    
      int rank, size;
    
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          assert(file >= 0);

          size_t dset_size = hdf5::dataset_num_elements(file, hdf5::cell_attribute_path(attr_name_space, pop_name, hdf5::CELL_INDEX));
          cell_index.resize(dset_size);
          ierr = hdf5::read<CELL_IDX_T> (file,
                                         hdf5::cell_attribute_path(attr_name_space, pop_name, hdf5::CELL_INDEX),
                                         0, dset_size,
                                         CELL_IDX_H5_NATIVE_T,
                                         cell_index, H5P_DEFAULT);
          assert(ierr >= 0);
          ierr = H5Fclose (file);
          assert(ierr == 0);

          // Ensure that every cell index is unique
          std::set<CELL_IDX_T> index_set;
          for (size_t i=0; i<cell_index.size(); i++)
            {
              index_set.insert(cell_index[i]);
            }
          assert(cell_index.size() == index_set.size());
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
     const string&        attr_name_space,
     const vector<CELL_IDX_T>&  cell_index
     )
    {
      herr_t ierr = 0;
      int srank, ssize; size_t rank, size;
    
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      assert(srank >= 0);
      assert(ssize > 0);
      
      rank = (size_t)srank;
      size = (size_t)ssize;
      
      hsize_t local_index_size = cell_index.size();

      std::vector<size_t> index_size_vector;
      index_size_vector.resize(size);
      ierr = MPI_Allgather(&local_index_size, 1, MPI_SIZE_T, &index_size_vector[0], 1, MPI_SIZE_T, comm);
      assert(ierr == MPI_SUCCESS);

      hsize_t local_index_start = 0;
      for (size_t i=0; i<rank; i++)
        {
          local_index_start = local_index_start + index_size_vector[i];
        }
      hsize_t global_index_size = 0;
      for (size_t i=0; i<size; i++)
        {
          global_index_size = global_index_size + index_size_vector[i];
        }

      ierr = create_cell_index(comm, file_name, pop_name, attr_name_space);
      
      hid_t file = hdf5::open_file(comm, file_name, true, true);
      assert(file >= 0);

      /* Create property list for collective dataset write. */
      hid_t wapl;
      wapl = H5Pcreate (H5P_DATASET_XFER);
      ierr = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);

      string path = hdf5::cell_attribute_path(attr_name_space, pop_name, hdf5::CELL_INDEX);
      
      hsize_t start = hdf5::dataset_num_elements(file, path);

      ierr = hdf5::write<CELL_IDX_T> (file, path,
                                      start+global_index_size, start+local_index_start, local_index_size,
                                      CELL_IDX_H5_NATIVE_T, cell_index, wapl);
      assert(ierr == 0);
      ierr = H5Fclose (file);
      assert(ierr == 0);

      return ierr;
    }

        
    herr_t link_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     const string&        attr_name
     )
    {
      herr_t ierr = 0;
      int srank, ssize; size_t rank, size;
    
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      assert(srank >= 0);
      assert(ssize > 0);
      
      rank = (size_t)srank;
      size = (size_t)ssize;

      string attr_path = hdf5::cell_attribute_path(attr_name_space, pop_name, attr_name);
      string attr_prefix = hdf5::cell_attribute_prefix(attr_name_space, pop_name);
      
      hid_t file = hdf5::open_file(comm, file_name, true, true);
      assert(file >= 0);
      
      hid_t dset = H5Dopen2(file, (attr_prefix + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      
      ierr = H5Olink(dset, file, (attr_path + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT, H5P_DEFAULT);
      assert(ierr >= 0);

      assert(H5Dclose(dset) >= 0);

      ierr = H5Fclose (file);
      assert(ierr == 0);

      return ierr;
    }



    
  }
}
