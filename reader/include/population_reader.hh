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

extern herr_t read_population_combos
(
 MPI_Comm                             comm,
 const std::string&                   file_name, 
 std::set< std::pair<pop_t,pop_t> >&  pop_pairs
 );

extern herr_t read_population_ranges
(
 MPI_Comm           comm,
 const std::string& file_name, 
 pop_range_map_t&   pop_ranges,
 std::vector<pop_range_t> &pop_vector,
 size_t &total_num_nodes
 );

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
