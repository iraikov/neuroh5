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
    int read_txt_projection (const string&          file_name,
                             const vector <size_t>& num_attrs,
                             vector<NODE_IDX_T>&    dst_idx,
                             vector<DST_PTR_T>&     src_idx_ptr,
                             vector<NODE_IDX_T>&    src_idx,
                             neuroh5::data::AttrVal& attrs);
  }
}
