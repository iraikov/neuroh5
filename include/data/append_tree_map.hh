
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_tree_map.hh
///
///  Populates a mapping between ranks and tree values.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#ifndef APPEND_TREE_MAP_HH
#define APPEND_TREE_MAP_HH

#include <vector>
#include <map>

#include "neuroh5_types.hh"

namespace neuroh5
{

  namespace data
  {
    void append_tree_map
    (NamedAttrMap&       attr_values,
     map<CELL_IDX_T, neurotree_t> &rank_tree_map);
    
  }
}
#endif
