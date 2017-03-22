// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_neighborhood.hh
///
///  Calculate vertex neighborhood set.
///
///  Copyright (C) 2016-2017 Project Neurograph.
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

    int vertex_neighbors (MPI_Comm comm,
                          const size_t global_num_nodes,
                          const map<NODE_IDX_T, vector<NODE_IDX_T> > &src_edge_map,
                          const map<NODE_IDX_T, vector<NODE_IDX_T> > &dst_edge_map,
                          vector< uint32_t > &degree_vector)
  }
}

#endif
