// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_reader.cc
///
///  Top-level functions for partitioning graphs in DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#ifndef GRAPH_PARTS_HH
#define GRAPH_PARTS_HH

#include "ngh5types.hh"

#include <mpi.h>
#include <hdf5.h>
#include <parmetis.h>

#include <map>
#include <vector>

namespace ngh5
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

  int partition_graph
  (
   MPI_Comm comm,
   const std::string& file_name,
   const std::vector<std::string> prj_names,
   const size_t Nparts,
   const size_t io_size,
   std::vector< idx_t > &parts
   );

  
}

#endif
