// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_txt_projection.cc
///
///  Read projection in text format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <string>
#include <vector>

#include "neuroh5_types.hh"
#include "attr_val.hh"

namespace neuroh5
{
  namespace io
  {
    int read_txt_projection (const std::string&          file_name,
                             const std::vector <size_t>& num_attrs,
                             std::vector<NODE_IDX_T>&    dst,
                             std::vector<NODE_IDX_T>&    src,
                             data::AttrVal&      attrs);
  }
}
