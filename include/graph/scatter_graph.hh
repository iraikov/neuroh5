// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_graph.hh
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#ifndef GRAPH_SCATTER_HH
#define GRAPH_SCATTER_HH

#include "neuroh5_types.hh"
#include "read_graph.hh"

#include <mpi.h>

#include <map>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    
    /// @brief Reads the edges of the given projections and scatters to all
    ///        ranks
    ///
    /// @param comm          MPI communicator
    ///
    /// @param scatter_type  Enumerate edges by destination (default) or source          
    ///
    /// @param file_name     Input file name
    ///
    /// @param io_size       Number of I/O ranks (those ranks that conduct I/O
    ///                      operations)
    ///
    /// @param opt_attrs     If true, read edge attributes
    ///
    /// @param prj_names     Vector of projection names to be read
    ///
    /// @param prj_vector    Vector of maps where the key is destination index,
    ///                      and the value is a vector of source indices and
    ///                      optional edge attributes, to be filled by this
    ///                      procedure
    ///
    /// @param total_num_nodes  Updated with the total number of nodes
    ///                         (vertices) in the graph
    ///
    /// @return              HDF5 error code
    int scatter_graph
    (
     MPI_Comm                           all_comm,
     const EdgeMapType                  edge_map_type,
     const std::string&                 file_name,
     const int                          io_size,
     const bool                         opt_attrs,
     const std::vector<std::string>&    prj_names,
     // A vector that maps nodes to compute ranks
     const std::map<NODE_IDX_T, rank_t>&  node_rank_map,
     std::vector < edge_map_t >& prj_vector,
     std::vector < std::vector <std::vector<std::string>> >& edge_attr_names_vector,
     size_t                            &total_num_nodes,
     size_t                            &local_num_edges,
     size_t                            &total_num_edges
     );
  }
}

#endif
