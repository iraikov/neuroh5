// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_degree_map.hh
///
///  Calculate vertex degrees as well as global maximum and minimum degree.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef VERTEX_DEGREE_HH
#define VERTEX_DEGREE_HH

#include "model_types.hh"

#include <mpi.h>
#include <vector>
#include <map>

namespace ngh5
{
  namespace graph
  {
    int vertex_degree (MPI_Comm comm,
                       const size_t total_num_nodes,
                       const std::map<NODE_IDX_T, std::vector<NODE_IDX_T> > &edge_map,
                       std::vector< uint32_t > &degree_map);
  }
}

#endif
