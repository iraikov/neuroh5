// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_reader.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#ifndef GRAPH_SCATTER_HH
#define GRAPH_SCATTER_HH

#include "ngh5types.hh"

#include <mpi.h>
#include <hdf5.h>

#include <map>
#include <vector>

namespace ngh5
{
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
  /// @param prj_list      Vector of projection tuples, to be filled with edge information by this procedure
  ///
  /// @param total_num_nodes  Updated with the total number of nodes (vertices) in the graph
  ///
  /// @param local_prj_num_edges  Updated with the number of edges in the graph read by the current (local) rank
  ///
  /// @param total_prj_num_edges  Updated with the total number of edges in the graph
  ///
  /// @return              HDF5 error code

  int read_graph
  (
   MPI_Comm comm,
   const std::string& file_name,
   const bool opt_attrs,
   const std::vector<std::string> prj_names,
   std::vector<prj_tuple_t> &prj_list,
   size_t &total_num_nodes,
   size_t &local_prj_num_edges,
   size_t &total_prj_num_edges
   );

  /// @brief Reads the edges of the given projections and scatters to all ranks, arranged by destination.
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param io_size       Number of I/O ranks (those ranks that conduct I/O operations)
  ///
  /// @param opt_attrs     If true, read edge attributes
  ///
  /// @param prj_names     Vector of projection names to be read
  ///
  /// @param prj_vector    Vector of maps where the key is destination index, and the value is a vector of source indices and optional edge attributes, to be filled by this procedure
  ///
  /// @param total_num_nodes  Updated with the total number of nodes (vertices) in the graph
  ///
  /// @param local_prj_num_edges  Updated with the number of edges in the graph read by the current (local) rank
  ///
  /// @param total_prj_num_edges  Updated with the total number of edges in the graph
  ///
  /// @return              HDF5 error code
  int scatter_graph
  (
   MPI_Comm all_comm,
   const std::string& file_name,
   const int io_size,
   const bool opt_attrs,
   const std::vector<std::string> prj_names,
   // A vector that maps nodes to compute ranks
   const std::vector<rank_t> node_rank_vector,
   std::vector < edge_map_t > & prj_vector,
   size_t &total_num_nodes,
   size_t &local_prj_num_edges,
   size_t &total_prj_num_edges
   );

  
}

#endif
