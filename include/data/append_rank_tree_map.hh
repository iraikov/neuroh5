
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_tree_map.cc
///
///  Populates a mapping between ranks and tree values.
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
    void append_rank_tree_map
    (NamedAttrMap&       attr_values,
     const map<CELL_IDX_T, rank_t>& node_rank_map,
     map <rank_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map);
    
  }
}
