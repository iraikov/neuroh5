// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_populations
///
///  Functions for reading population information and validating the
///  source and destination indices of edges.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================
#ifndef CELL_POPULATIONS_HH
#define CELL_POPULATIONS_HH

#include <mpi.h>
#include <vector>
#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace cell
  {

    /// @brief Reads the valid combinations of source/destination populations.
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param pop_pairs     Set of source/destination pairs, filled by this
    ///                      procedure
    ///
    /// @return              HDF5 error code
    extern herr_t read_population_combos
    (
     MPI_Comm                                           comm,
     const std::string&                                 file_name,
     std::set< std::pair<pop_t,pop_t> >&  pop_pairs
     );

    /// @brief Reads the id ranges of each population
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param pop_ranges    Map where the key is the starting index of a
    ///                      population, and the value is the number of nodes
    ///                      (vertices) and population index, filled by this
    ///                      procedure
    ///
    /// @param pop_vector    Vector of tuples <start,count,population index>
    ///                      for each population, filled by this procedure
    ///
    /// @param total_num_nodes    Updated with total number of nodes (vertices)
    ///
    /// @return              HDF5 error code
    extern herr_t read_population_ranges
    (
     MPI_Comm                         comm,
     const std::string&               file_name,
     pop_range_map_t&          pop_ranges,
     std::vector<pop_range_t>& pop_vector,
     size_t&                          total_num_nodes
     );

    extern herr_t read_population_labels
    (
     MPI_Comm                         comm,
     const std::string&               file_name,
     std::vector< std::pair<pop_t, std::string> > & pop_labels
     );

    
    herr_t read_population_names
    (
     MPI_Comm             comm,
     hid_t                file,
     vector<string>&      pop_names
     );
  }
}

#endif
