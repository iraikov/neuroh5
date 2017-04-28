// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph.hh
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef READ_GRAPH_HH
#define READ_GRAPH_HH

#include "model_types.hh"

#include <mpi.h>

#include <vector>
#include <map>

namespace ngh5
{
  namespace graph
  {
        
    enum EdgeMapType
      {
        EdgeMapDst,
        EdgeMapSrc
      };


    extern int append_rank_edge_map
    (
     const NODE_IDX_T&                 dst_start,
     const NODE_IDX_T&                 src_start,
     const std::vector<DST_BLK_PTR_T>& dst_blk_ptr,
     const std::vector<NODE_IDX_T>&    dst_idx,
     const std::vector<DST_PTR_T>&     dst_ptr,
     const std::vector<NODE_IDX_T>&    src_idx,
     const model::EdgeNamedAttr&       edge_attr_values,
     const std::map<NODE_IDX_T, model::rank_t>& node_rank_map,
     size_t&                           num_edges,
     model::rank_edge_map_t &          rank_edge_map,
     EdgeMapType                       edge_map_type
     );

    extern int append_edge_map
    (
     const NODE_IDX_T&                 dst_start,
     const NODE_IDX_T&                 src_start,
     const std::vector<DST_BLK_PTR_T>& dst_blk_ptr,
     const std::vector<NODE_IDX_T>&    dst_idx,
     const std::vector<DST_PTR_T>&     dst_ptr,
     const std::vector<NODE_IDX_T>&    src_idx,
     const model::EdgeNamedAttr&       edge_attr_values,
     size_t&                           num_edges,
     model::edge_map_t &               edge_map,
     EdgeMapType                       edge_map_type
     );

    extern int append_prj_list
    (
     const NODE_IDX_T&                                  dst_start,
     const NODE_IDX_T&                                  src_start,
     const std::vector<DST_BLK_PTR_T>&                  dst_blk_ptr,
     const std::vector<NODE_IDX_T>&                     dst_idx,
     const std::vector<DST_PTR_T>&                      dst_ptr,
     const std::vector<NODE_IDX_T>&                     src_idx,
     const model::EdgeNamedAttr&                        edge_attr_values,
     size_t&                                            num_edges,
     std::vector<model::prj_tuple_t>&                   prj_list
     );

    /// @brief Reads the edges of the given projections
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param opt_attrs     If true, read edge attributes
    ///
    /// @param prj_names     Vector of projection names to be read
    ///
    /// @param prj_list      Vector of projection tuples, to be filled with
    ///                      edge information by this procedure
    ///
    /// @param total_num_nodes  Updated with the total number of nodes
    ///                         (vertices) in the graph
    ///
    /// @param local_prj_num_edges  Updated with the number of edges in the
    ///                             graph read by the current (local) rank
    ///
    /// @param total_prj_num_edges  Updated with the total number of edges in
    ///                             the graph
    ///
    /// @return              HDF5 error code

    extern int read_graph
    (
     MPI_Comm                         comm,
     const std::string&               file_name,
     const bool                       opt_attrs,
     const std::vector<std::string>   prj_names,
     std::vector<model::prj_tuple_t>& prj_list,
     size_t&                          total_num_nodes,
     size_t&                          local_prj_num_edges,
     size_t&                          total_prj_num_edges
     );
  }
}

#endif
