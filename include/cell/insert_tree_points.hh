// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file insert_tree_points.hh
///
///  Insert points into tree structure.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include <algorithm>
#include <set>

#include "neuroh5_types.hh"

namespace neuroh5
{

  namespace cell
  {
    
    void insert_tree_points(const neurotree_t& src_tree, neurotree_t& dst_tree);
  }
}
