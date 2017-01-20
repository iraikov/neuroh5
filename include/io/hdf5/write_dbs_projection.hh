// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file write_dbs_projection.cc
///
///  Functions for writing edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "ngh5_types.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>


namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      
      
      herr_t write_dbs_projection
      (
       MPI_Comm                   comm,
       const std::string&         file_name,
       const std::string&         proj_name,
       const NODE_IDX_T&          dst_start,
       const NODE_IDX_T&          src_start,
       const DST_BLK_PTR_T&       block_base,
       const DST_PTR_T&           edge_base,
       const vector<NODE_IDX_T>&  dst_idx,
       const vector<NODE_IDX_T>&  src_idx
       );
    }
  }
}
