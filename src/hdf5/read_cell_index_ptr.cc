// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_cell_index_ptr
///
///  Functions for reading NeuroH5 cell attribute index and pointer.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include <hdf5.h>

#include <iostream>
#include <sstream>
#include <string>

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "exists_group.hh"
#include "path_names.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    

    herr_t read_cell_index
    (
     const hid_t&              loc,
     const std::string&        path,
     std::vector<CELL_IDX_T> & index,
     bool collective
     )
    {
      herr_t status = 0;


      status = exists_group (loc, path.c_str());
      throw_assert(status > 0,
                   "read_cell_index_ptr: group " << path << " does not exist");
      
      hsize_t dset_size = dataset_num_elements (loc, path + "/" + CELL_INDEX);
      
      if (dset_size > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
          if (collective)
            {
              status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              throw_assert(status >= 0,
                           "read_cell_index_ptr: error in H5Pset_dxpl_mpio");
            }
          
          string index_path = path + "/" + CELL_INDEX;

          // read index
          index.resize(dset_size);
          status = read<NODE_IDX_T> (loc, index_path, 0, dset_size,
                                     NODE_IDX_H5_NATIVE_T, index, rapl);

          status = H5Pclose(rapl);
          throw_assert(status == 0, "read_cell_index_ptr: unable to close property list");
          
        }
      
      return status;
    }


    herr_t read_cell_index_ptr
    (
     const hid_t&              loc,
     const std::string&        path,
     std::vector<CELL_IDX_T> & index,
     std::vector<ATTR_PTR_T> & ptr,
     bool collective
     )
    {
      herr_t status = 0;


      status = exists_group (loc, path.c_str());
      throw_assert(status > 0,
                   "read_cell_index_ptr: group " << path << " does not exist");
      
      hsize_t dset_size = dataset_num_elements (loc, path + "/" + CELL_INDEX);
      
      if (dset_size > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
          if (collective)
            {
              status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              throw_assert(status >= 0,
                           "read_cell_index_ptr: error in H5Pset_dxpl_mpio");
            }
          
          string index_path = path + "/" + CELL_INDEX;
          string ptr_path = path + "/" + ATTR_PTR;

          // read index
          index.resize(dset_size);
          status = read<NODE_IDX_T> (loc, index_path, 0, dset_size,
                                     NODE_IDX_H5_NATIVE_T, index, rapl);

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
          throw_assert(status == 0, "read_cell_index_ptr: unable to close property list");
          
        }
      
      return status;
    }
  }
}
