// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file balance_graph_indegree.hh
///
///  Function definitions for balancing graphs.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#ifndef BALANCE_GRAPH_INDEGREE_HH
#define BALANCE_GRAPH_INDEGREE_HH

#include "model_types.hh"


#include <mpi.h>
#include <hdf5.h>

#include <map>
#include <vector>

namespace ngh5
{
  namespace graph
  {
  /// @brief Partitions the given projections to minimize the edge cut for each MPI rank
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param prj_names     Vector of projection names to be read
  ///
  /// @param io_size       Number of I/O ranks (those ranks that conduct I/O operations)
  ///
  /// @param Nparts        Number of partitions
  ///
  /// @param parts         Updated with node vectors containing the partitioner output
  ///                      (assignment of graph nodes to MPI ranks)
  ///
  /// @return              HDF5 error code

    int balance_graph_indegree
    (
     MPI_Comm comm,
     const std::string& input_file_name,
     const std::vector<std::string> prj_names,
     const size_t io_size,
     const size_t Nparts,
     std::vector<NODE_IDX_T> &parts,
     std::vector<double> &part_weights
     );

  }
  
}

#endif
