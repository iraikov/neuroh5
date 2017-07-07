// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_layer_swc
///
///  Definition for layer SWC import routine.
///
///  Copyright (C) 2016 Project NeuroH5.
//==============================================================================
#ifndef READ_LAYER_SWC_HH
#define READ_LAYER_SWC_HH

#include <vector>
#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace io
  {
    int read_layer_swc
    (
     const std::string& file_name,
     const CELL_IDX_T gid,
     const int id_offset,
     const int layer_offset,
     const SWC_TYPE_T swc_type,
     const bool split_layers,
     std::vector<neuroh5::neurotree_t> &tree_list
     );

    int read_swc
    (
     const std::string& file_name,
     const CELL_IDX_T gid,
     const int id_offset,
     vector<neurotree_t> &tree_list
     );
  }
    
}

#endif
