// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection.hh
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef READ_PROJECTION_HH
#define READ_PROJECTION_HH

#include "neuroh5_types.hh"

#include <mpi.h>

#include <map>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace graph
  {

    /// @brief Reads the projections
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param src_pop_name  Source population name
    ///
    /// @param dst_pop_name  Destination population name
    ///
    /// @param dst_start     Updated with global starting index of destination
    ///                      population
    ///
    /// @param src_start     Updated with global starting index of source
    ///                      population
    ///
    /// @param nedges        Total number of edges in the projection
    ///
    /// @param block_base    Global index of the first block read by this task
    ///
    /// @param edge_base     Global index of the first edge read by this task
    ///
    /// @param dst_blk_ptr   Destination Block Pointer (pointer to Destination
    ///                      Pointer for blocks of connectivity)
    ///
    /// @param dst_idx       Destination Index (starting destination index for
    ///                      each block)
    ///
    /// @param dst_ptr       Destination Pointer (pointer to Source Index where
    ///                      the source indices for a given destination can be
    ///                      located)
    ///
    /// @param src_idx       Source Index (source indices of edges)
    ///
    /// @return              HDF5 error code
    extern herr_t read_projection
    (
     MPI_Comm                        comm,
     const std::string&              file_name,
     const pop_range_map_t&          pop_ranges,
     const std::set< std::pair<pop_t, pop_t> >& pop_pairs,
     const std::string&              src_pop_name,
     const std::string&              dst_pop_name,
     const NODE_IDX_T&               dst_start,
     const NODE_IDX_T&               src_start,
     const vector<string>&           edge_attr_name_spaces,
     vector<prj_tuple_t>&            prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                         local_num_edges,
     size_t&                         total_num_edges,
     size_t                          offset = 0,
     size_t                          numitems = 0,
     bool collective = true
     );

    extern herr_t read_projection_serial
    (
     const std::string&              file_name,
     const pop_range_map_t&          pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&              src_pop_name,
     const std::string&              dst_pop_name,
     const NODE_IDX_T&               dst_start,
     const NODE_IDX_T&               src_start,
     const vector<string>&           attr_namespaces,
     vector<prj_tuple_t>&            prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                         total_num_edges,
     size_t                          offset = 0,
     size_t                          numitems = 0
     );
  }
}

#endif
