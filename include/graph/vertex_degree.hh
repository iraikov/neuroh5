// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_degree.hh
///
///  Calculate vertex degrees as well as global maximum and minimum degree.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef VERTEX_DEGREE_HH
#define VERTEX_DEGREE_HH

#include "neuroh5_types.hh"

#include <mpi.h>
#include <vector>
#include <map>

namespace neuroh5
{
  namespace graph
  {
    int vertex_degree (MPI_Comm comm,
                       const size_t global_num_nodes,
                       const vector < edge_map_t >& prj_vector,
                       vector < map< NODE_IDX_T, uint32_t > > &degree_maps);
  }
}

#endif
