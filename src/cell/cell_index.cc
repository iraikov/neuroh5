// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_tree_index.cc
///
///  Functions for reading tree indices from an HDF5 file.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <hdf5.h>

#include <cstring>
#include <vector>

#include "neurotrees_types.hh"
#include "dataset_num_elements.hh"
#include "mpi_bcast_string_vector.hh"
#include "hdf5_path_names.hh"
#include "hdf5_types.hh"
#include "hdf5_read_template.hh"

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neurotrees
{

  
  //////////////////////////////////////////////////////////////////////////
  herr_t read_tree_index
  (
   MPI_Comm             comm,
   const string&        file_name,
   const string&        pop_name,
   vector<CELL_IDX_T>&  tree_index
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

        size_t dset_size = dataset_num_elements(comm, file, cell_attribute_path(TREES, pop_name, ATTR_PTR));
        tree_index.resize(dset_size-1);
        ierr = hdf5_read<CELL_IDX_T> (file,
                                      cell_attribute_path(TREES, pop_name, TREE_ID),
                                      0, dset_size-1,
                                      CELL_IDX_H5_NATIVE_T,
                                      tree_index, H5P_DEFAULT);
        assert(ierr >= 0);
        ierr = H5Fclose (file);
        assert(ierr == 0);
      }
    uint32_t numitems = tree_index.size();
    ierr = MPI_Bcast(&numitems, 1, MPI_UINT32_T, 0, comm);
    assert(ierr == MPI_SUCCESS);
    
    tree_index.resize(numitems);
    ierr = MPI_Bcast(&(tree_index[0]), numitems, MPI_CELL_IDX_T, 0, comm);
    assert(ierr == MPI_SUCCESS);
    
    return ierr;
  }

}
