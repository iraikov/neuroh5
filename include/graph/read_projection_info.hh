// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_info.hh
///
///  Top-level functions for reading subsets of graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#ifndef READ_PROJECTION_INFO_HH
#define READ_PROJECTION_INFO_HH

#include "neuroh5_types.hh"

#include <mpi.h>

#include <vector>
#include <map>
#include <algorithm>
#include <utility>

namespace neuroh5
{
  namespace graph
  {


    herr_t read_projection_info
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const vector<string>& edge_attr_name_spaces,
     const bool                 read_node_index,
     const pop_range_map_t&     pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     vector< pair<string, string> >& prj_names,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     std::vector<std::vector<NODE_IDX_T>>& prj_node_index
     
     );

    herr_t has_projection
    (
     MPI_Comm                      comm,
     const string&                 file_name,
     const string&                 src_pop_name,
     const string&                 dst_pop_name,
     bool &has_projection
     );

  }
}

#endif
