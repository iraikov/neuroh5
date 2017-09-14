
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_prj_map.cc
///
///  Populates a list of projections with edge values. 
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
    int append_prj_list
    (
     const NODE_IDX_T&                                  dst_start,
     const NODE_IDX_T&                                  src_start,
     const std::vector<DST_BLK_PTR_T>&                  dst_blk_ptr,
     const std::vector<NODE_IDX_T>&                     dst_idx,
     const std::vector<DST_PTR_T>&                      dst_ptr,
     const std::vector<NODE_IDX_T>&                     src_idx,
     const std::map<std::string, data::NamedAttrVal>&   edge_attr_map,
     size_t&                                            num_edges,
     std::vector<prj_tuple_t>&                          prj_list
     );
  }
}


