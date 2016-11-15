// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file dbs_edge_reader.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef DBS_GRAPH_READER_HH
#define DBS_GRAPH_READER_HH

#include "ngh5types.hh"

#include "mpi.h"

#include <map>
#include <string>
#include <vector>

namespace ngh5
{

  /// @brief Reads the names of projections
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param prj_vector    Vector of projection names
  ///
  /// @return              HDF5 error code
  extern herr_t read_projection_names
  (
   MPI_Comm                 comm,
   const std::string&       file_name,
   std::vector<std::string> &prj_vector
   );


  /// @brief Reads the projections
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param proj_name     Projection name
  ///
  /// @param pop_vector    Population ranges
  ///
  /// @param dst_start     Updated with global starting index of destination population
  ///
  /// @param src_start     Updated with global starting index of source population
  ///
  /// @param nedges        Total number of edges in the projection
  ///
  /// @param block_base    Global index of the first block read by this task
  ///
  /// @param edge_base     Global index of the first edge read by this task
  ///
  /// @param dst_blk_ptr   Destination Block Pointer (pointer to Destination Pointer for blocks of connectivity)
  ///
  /// @param dst_idx       Destination Index (starting destination index for each block)
  ///
  /// @param dst_ptr       Destination Pointer (pointer to Source Index where the source indices for a given destination can be located)
  ///
  /// @param src_idx       Source Index (source indices of edges)
  ///
  /// @return              HDF5 error code
  extern herr_t read_dbs_projection
  (
   MPI_Comm                        comm,
   const std::string&              file_name,
   const std::string&              proj_name,
   const std::vector<pop_range_t>& pop_vector,
   NODE_IDX_T&                     dst_start,
   NODE_IDX_T&                     src_start,
   uint64_t&                       nedges,
   DST_BLK_PTR_T&                  block_base,
   DST_PTR_T&                      edge_base,
   std::vector<DST_BLK_PTR_T>&     dst_blk_ptr,
   std::vector<NODE_IDX_T>&        dst_idx,
   std::vector<DST_PTR_T>&         dst_ptr,
   std::vector<NODE_IDX_T>&        src_idx
   );
}

#endif
