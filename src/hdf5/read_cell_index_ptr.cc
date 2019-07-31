// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_cell_index_ptr
///
///  Functions for reading NeuroH5 cell attribute index and pointer.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================

#include <hdf5.h>

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "throw_assert.hh"

#include <iostream>
#include <sstream>
#include <string>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    

    herr_t read_cell_index_ptr
    (
     MPI_Comm                  comm,
     const hid_t&              loc,
     const std::string&        path,
     const CELL_IDX_T          pop_start,
     std::vector<CELL_IDX_T> & index,
     std::vector<ATTR_PTR_T> & ptr
     )
    {
      herr_t status = 0;

      int size, rank;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      status = exists_group (loc, path.c_str());
      throw_assert(status > 0,
                   "read_cell_index_ptr: group does not exist");
      
      hsize_t dset_size = dataset_num_elements (loc, path + "/" + CELL_INDEX);
      
      if (dset_size > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
          
          string index_path = path + "/" + CELL_INDEX;
          string ptr_path = path + "/" + ATTR_PTR;

          // read index
          index.resize(dset_size);
          status = read<NODE_IDX_T> (loc, index_path, 0, dset_size,
                                     NODE_IDX_H5_NATIVE_T, index, rapl);
          for (size_t i=0; i<index.size(); i++)
            {
              index[i] += pop_start;
            }

          // read pointer
          status = exists_dataset (loc, ptr_path);
          if (status > 0)
            {
              ptr.resize(dset_size+1);
              status = read<ATTR_PTR_T> (loc, ptr_path, 0, dset_size+1,
                                         ATTR_PTR_H5_NATIVE_T, ptr, rapl);
              throw_assert (status >= 0,
                            "read_cell_index_ptr: error in read");
            }
          
          status = H5Pclose(rapl);
          assert(status == 0);
          
        }
      
      return status;
    }
  }
}
