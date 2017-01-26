// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file write_graph.hh
///
///  Top-level functions for writing graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#ifndef WRITE_GRAPH_HH
#define WRITE_GRAPH_HH

#include "model_types.hh"
#include "attr_map.hh"

#include <mpi.h>
#include <vector>

namespace ngh5
{
  namespace graph
  {
    

    int write_graph
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    prj_name,
     const bool            opt_attrs,
     const std::vector<NODE_IDX_T> edges,
     const model::NamedAttrMap& edge_attrs
     );
  }
}

#endif
