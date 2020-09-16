
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_edge_map_selection.hh
///
///  Populates a mapping between node indices and edge values.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#ifndef APPEND_EDGE_MAP_SELECTION_HH
#define APPEND_EDGE_MAP_SELECTION_HH

#include <vector>
#include <map>

#include "neuroh5_types.hh"

namespace neuroh5
{

  namespace data
  {
    int append_edge_map_selection
    (
     const NODE_IDX_T&                       dst_start,
     const NODE_IDX_T&                       src_start,
     const std::vector<NODE_IDX_T>&          selection_dst_idx,
     const std::vector<DST_PTR_T>&           selection_dst_ptr,
     const std::vector<NODE_IDX_T>&          src_idx,
     const vector<string>&                   attr_namespaces,
     const map<string, data::NamedAttrVal>&  edge_attr_map,
     size_t&                                 num_edges,
     edge_map_t &                            edge_map,
     EdgeMapType                             edge_map_type
     );
  }
}

#endif
