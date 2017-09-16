// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph.hh
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef READ_GRAPH_HH
#define READ_GRAPH_HH

#include "neuroh5_types.hh"

#include <mpi.h>

#include <vector>
#include <map>
#include <utility>

namespace neuroh5
{
  namespace graph
  {


    /// @brief Reads the edges of the given projections
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param attr_namespaces     Vector of edge attribute namespaces to read
    ///
    /// @param prj_names     Vector of <src, dst> projections to be read
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
     const std::vector< std::string >& edge_attr_name_spaces,
     const std::vector< std::pair<std::string, std::string> >& prj_names,
     std::vector<prj_tuple_t>&        prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                          total_num_nodes,
     size_t&                          local_prj_num_edges,
     size_t&                          total_prj_num_edges
     );

    extern int read_graph_serial
    (
     const std::string&               file_name,
     const std::vector< std::string >& edge_attr_name_spaces,
     const std::vector< std::pair<std::string, std::string> >&  prj_names,
     std::vector<prj_tuple_t>&        prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                          total_num_nodes,
     size_t&                          total_prj_num_edges
     );
  }
}

#endif
