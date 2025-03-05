// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file node_rank_map.hh
///
///  Function for creating a mapping of graph nodes to MPI ranks.
///
///  Copyright (C) 2025 Project NeuroH5.
//==============================================================================

#ifndef NODE_RANK_MAP_HH
#define NODE_RANK_MAP_HH

#include <mpi.h>

#include <vector>
#include <map>
#include <algorithm>
#include <cassert>

#include "throw_assert.hh"

using namespace std;


namespace neuroh5
{

  namespace mpi
  {


    void compute_node_rank_map
    (
     size_t num_ranks,
     size_t num_nodes,
     size_t &total_num_nodes,
     map<NODE_IDX_T, rank_t> &node_rank_map
     );
  }
}

#endif
    
