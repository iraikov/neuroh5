// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file population_reader.cc
///
///  Functions for reading population information and validating the
///  source and destination indices of edges.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef POPULATION_READER_HH
#define POPULATION_READER_HH

#include "ngh5types.hh"

#include <map>
#include <set>
#include <map>
#include <vector>

#include "hdf5.h"

namespace ngh5
{

  /// @brief Reads the valid combinations of source/destination populations.
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param pop_pairs     Set of source/destination pairs, filled by this procedure
  ///
  /// @return              HDF5 error code
  extern herr_t read_population_combos
  (
   MPI_Comm                             comm,
   const std::string&                   file_name, 
   std::set< std::pair<pop_t,pop_t> >&  pop_pairs
   );

  /// @brief Reads the id ranges of each population
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param pop_ranges    Map where the key is the starting index of a population, and the value is the number of nodes (vertices) and population index, filled by this procedure
  ///
  /// @param pop_vector    Vector of tuples <start,count,population index> for each population, filled by this procedure
  ///
  /// @param total_num_nodes    Updated with total number of nodes (vertices)
  ///
  /// @return              HDF5 error code
  extern herr_t read_population_ranges
  (
   MPI_Comm           comm,
   const std::string& file_name, 
   pop_range_map_t&   pop_ranges,
   std::vector<pop_range_t> &pop_vector,
   size_t &total_num_nodes
   );

  /// @brief Validates that each edge in a projection has source and destination indices that are within the population ranges defined for that projection
  ///
  /// @param dst_start     Updated with global starting index of destination population
  ///
  /// @param src_start     Updated with global starting index of source population
  ///
  /// @param dst_blk_ptr   Destination Block Pointer (pointer to Destination Pointer for blocks of connectivity)
  ///
  /// @param dst_idx       Destination Index (starting destination index for each block)
  ///
  /// @param dst_ptr       Destination Pointer (pointer to Source Index where the source indices for a given destination can be located)
  ///
  /// @param src_idx       Source Index (source indices of edges)
  ///
  /// @param pop_ranges    Map where the key is the starting index of a population, and the value is the number of nodes (vertices) and population index, filled by this procedure
  ///
  /// @param pop_pairs     Set of source/destination pairs, filled by this procedure
  ///
  /// @return              True if the edges are valid, false otherwise
  extern bool validate_edge_list
  (
   NODE_IDX_T&         dst_start,
   NODE_IDX_T&         src_start,
   std::vector<DST_BLK_PTR_T>&  dst_blk_ptr,
   std::vector<NODE_IDX_T>& dst_idx,
   std::vector<DST_PTR_T>&  dst_ptr,
   std::vector<NODE_IDX_T>& src_idx,
   const pop_range_map_t&           pop_ranges,
   const std::set< std::pair<pop_t, pop_t> >& pop_pairs
   );

}

#endif
