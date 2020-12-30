
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_edge_map.cc
///
///  Populates a mapping between ranks and edge values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef APPEND_RANK_EDGE_MAP_HH
#define APPEND_RANK_EDGE_MAP_HH

#include <vector>
#include <map>

#include "neuroh5_types.hh"

namespace neuroh5
{

  namespace data
  {

    int append_rank_edge_map
    (
     const size_t                                initial_rank,
     const size_t                                num_ranks,
     const NODE_IDX_T&                           dst_start,
     const NODE_IDX_T&                           src_start,
     const std::vector<DST_BLK_PTR_T>&           dst_blk_ptr,
     const std::vector<NODE_IDX_T>&              dst_idx,
     const std::vector<DST_PTR_T>&               dst_ptr,
     const std::vector<NODE_IDX_T>&              src_idx,
     const vector<string>&                       attr_namespaces,
     const std::map<string, data::NamedAttrVal>& edge_attr_map,
     const node_rank_map_t&                      node_rank_map,
     size_t&                                     num_edges,
     rank_edge_map_t &                           rank_edge_map,
     EdgeMapType                                 edge_map_type
     );
  }
}
#endif
