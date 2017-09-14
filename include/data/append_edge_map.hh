
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_edge_map.cc
///
///  Populates a mapping between node indices and edge values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <vector>
#include <map>

#include "neuroh5_types.hh"

namespace neuroh5
{

  namespace data
  {
    int append_edge_map
    (
     const NODE_IDX_T&                 dst_start,
     const NODE_IDX_T&                 src_start,
     const std::vector<DST_BLK_PTR_T>& dst_blk_ptr,
     const std::vector<NODE_IDX_T>&    dst_idx,
     const std::vector<DST_PTR_T>&     dst_ptr,
     const std::vector<NODE_IDX_T>&    src_idx,
     const data::NamedAttrVal&         edge_attr_values,
     size_t&                           num_edges,
     edge_map_t &                      edge_map,
     EdgeMapType                       edge_map_type
     );
  }
}


    extern int append_prj_list
    (
     const NODE_IDX_T&                                  dst_start,
     const NODE_IDX_T&                                  src_start,
     const std::vector<DST_BLK_PTR_T>&                  dst_blk_ptr,
     const std::vector<NODE_IDX_T>&                     dst_idx,
     const std::vector<DST_PTR_T>&                      dst_ptr,
     const std::vector<NODE_IDX_T>&                     src_idx,
     const data::NamedAttrVal&                          edge_attr_values,
     size_t&                                            num_edges,
     std::vector<prj_tuple_t>&                          prj_list
     );
