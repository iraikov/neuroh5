
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_attr.cc
///
///  Populates a mapping between node indices and attribute values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"

namespace neuroh5
{

  namespace data
  {
  
    void append_rank_attr_map
    (const NamedAttrMap   &attr_values,
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     map <rank_t, AttrMap> &rank_attr_map);

  }
  
}

