// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file reader.cc
///
///  Driver program for scatter_graph function.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include <string>
#include <vector>

#include "ngh5_types.hh"
#include "edge_attr.hh"

namespace ngh5
{
  namespace io
  {
    int read_txt_projection (const std::string&          file_name,
                             const std::vector <size_t>& num_attrs,
                             std::vector<NODE_IDX_T>&    dst,
                             std::vector<NODE_IDX_T>&    src,
                             ngh5::model::EdgeAttr&      attrs);
  }
}
