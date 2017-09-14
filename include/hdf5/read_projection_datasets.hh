// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_datasets.hh
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "neuroh5_types.hh"

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    
    /**************************************************************************
     * Read the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection_datasets
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     DST_BLK_PTR_T&             block_base,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t&                    total_num_edges,
     size_t                     offset,
     size_t                     numitems,
     bool collective = true
     );

    
    herr_t read_projection_datasets_serial
    (
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t&                    total_num_edges,
     size_t                     offset,
     size_t                     numitems
     );
  }
}


