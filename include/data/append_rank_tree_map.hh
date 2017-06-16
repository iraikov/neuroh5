
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
    (
     const size_t start,
     const size_t num_trees,
     const map<CELL_IDX_T, rank_t>& node_rank_map,
     const CELL_IDX_T pop_start,
     vector<SEC_PTR_T>& sec_ptr,
     vector<TOPO_PTR_T>& topo_ptr,
     vector<ATTR_PTR_T>& attr_ptr,
     vector<CELL_IDX_T>& all_gid_vector,
     vector<SECTION_IDX_T>& all_src_vector,
     vector<SECTION_IDX_T>& all_dst_vector,
     vector<SECTION_IDX_T>& all_sections,
     vector<COORD_T>& all_xcoords,
     vector<COORD_T>& all_ycoords,
     vector<COORD_T>& all_zcoords,
     vector<REALVAL_T>& all_radiuses,
     vector<LAYER_IDX_T>& all_layers,
     vector<PARENT_NODE_IDX_T>& all_parents,
     vector<SWC_TYPE_T>& all_swc_types,
     map <rank_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map
     );
    
  }
}
