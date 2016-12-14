// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file merge_edge_map.hh
///
///  Merge edges from multiple projections into a single edge map.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef MERGE_EDGE_MAP_HH
#define MERGE_EDGE_MAP_HH

#include "model_types.hh"

#include <vector>
#include <map>

namespace ngh5
{
  namespace graph
  {

    void merge_edge_map (const std::vector < edge_map_t > &prj_vector,
                         std::map<NODE_IDX_T, std::vector<NODE_IDX_T> > &edge_map);
  }
}

