// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_context.hh
///
///  Type definitions for a graph context object, which contains
///  metadata associated with the graph.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================
#ifndef GRAPH_CONTEXT_HH
#define GRAPH_CONTEXT_HH

#include "model_types.hh"
#include <mpi.h>

#include <map>
#include <tuple>
#include <vector>

namespace ngh5
{
  namespace model
  {

    struct GraphContext
    {
      // Communicators for all ranks and I/O ranks
      MPI_Comm all_comm;
      MPI_Comm io_comm;
      
      // The number of I/O ranks
      int io_size;
      
      // MPI datatypes for packing edge information before sending to all ranks
      MPI_Datatype header_type;
      MPI_Datatype size_type;
      
      // Specifies destination- or source- oriented in-memory edge representation
      EdgeMapType edge_map_type;
      
      map<NODE_IDX_T, model::rank_t>  node_rank_map;
      vector<model::pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,model::pop_t> > pop_ranges;
      set< pair<model::pop_t, model::pop_t> > pop_pairs;

    }

  }
}

#endif
