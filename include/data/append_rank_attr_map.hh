
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_attr.cc
///
///  Populates a mapping between node indices and attribute values.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#ifndef APPEND_RANK_ATTR_MAP_HH
#define APPEND_RANK_ATTR_MAP_HH

#include <vector>
#include <set>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "attr_val.hh"

namespace neuroh5
{

  namespace data
  {
  

    void append_rank_attr_map
    (
     const data::NamedAttrMap   &attr_values,
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     map <rank_t, data::AttrMap> &rank_attr_map);

    void append_rank_attr_map
    (
     const data::NamedAttrMap   &attr_values,
     const map<CELL_IDX_T, set <rank_t> > &node_rank_map,
     map <rank_t, data::AttrMap> &rank_attr_map);
  }
  
}

#endif
