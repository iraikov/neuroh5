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
    int read_graph_info
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const vector<string>& edge_attr_name_spaces,
     const bool            read_node_index, 
     vector< pair<string, string> >& prj_names,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     std::vector<std::vector<NODE_IDX_T>>& prj_node_index
     );
  }
}

#endif
