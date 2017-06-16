
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_tree_map.cc
///
///  Populates a mapping between ranks and tree values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <cassert>
#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"

using namespace std;

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
     map <rank_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map)
    {
      for (size_t i=0; i<num_trees; i++)
        {
          hsize_t topo_start = topo_ptr[i]-topo_ptr[0]; size_t topo_block = topo_ptr[i+1]-topo_ptr[0]-topo_start;

          vector<SECTION_IDX_T>::const_iterator src_first = all_src_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator src_last  = all_src_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_src_vector;
          tree_src_vector.insert(tree_src_vector.begin(),src_first,src_last);
        
          vector<SECTION_IDX_T>::const_iterator dst_first = all_dst_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator dst_last  = all_dst_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_dst_vector;
          tree_dst_vector.insert(tree_dst_vector.begin(),dst_first,dst_last);
        
          hsize_t sec_start = sec_ptr[i]-sec_ptr[0]; size_t sec_block = sec_ptr[i+1]-sec_ptr[0]-sec_start;
        
          vector<SECTION_IDX_T>::const_iterator sec_first = all_sections.begin() + sec_start;
          vector<SECTION_IDX_T>::const_iterator sec_last  = all_sections.begin() + sec_start + sec_block;
          vector<SECTION_IDX_T> tree_sections;
          tree_sections.insert(tree_sections.begin(),sec_first,sec_last);
        
          hsize_t attr_start = attr_ptr[i]-attr_ptr[0]; size_t attr_block = attr_ptr[i+1]-attr_ptr[0]-attr_start;

          vector<COORD_T>::const_iterator xcoords_first = all_xcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator xcoords_last  = all_xcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_xcoords;
          tree_xcoords.insert(tree_xcoords.begin(),xcoords_first,xcoords_last);

          vector<COORD_T>::const_iterator ycoords_first = all_ycoords.begin() + attr_start;
          vector<COORD_T>::const_iterator ycoords_last  = all_ycoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_ycoords;
          tree_ycoords.insert(tree_ycoords.begin(),ycoords_first,ycoords_last);

          vector<COORD_T>::const_iterator zcoords_first = all_zcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator zcoords_last  = all_zcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_zcoords;
          tree_zcoords.insert(tree_zcoords.begin(),zcoords_first,zcoords_last);

          vector<REALVAL_T>::const_iterator radiuses_first = all_radiuses.begin() + attr_start;
          vector<REALVAL_T>::const_iterator radiuses_last  = all_radiuses.begin() + attr_start + attr_block;
          vector<REALVAL_T> tree_radiuses;
          tree_radiuses.insert(tree_radiuses.begin(),radiuses_first,radiuses_last);

          vector<LAYER_IDX_T>::const_iterator layers_first = all_layers.begin() + attr_start;
          vector<LAYER_IDX_T>::const_iterator layers_last  = all_layers.begin() + attr_start + attr_block;
          vector<LAYER_IDX_T> tree_layers;
          tree_layers.insert(tree_layers.begin(),layers_first,layers_last);

          vector<PARENT_NODE_IDX_T>::const_iterator parents_first = all_parents.begin() + attr_start;
          vector<PARENT_NODE_IDX_T>::const_iterator parents_last  = all_parents.begin() + attr_start + attr_block;
          vector<PARENT_NODE_IDX_T> tree_parents;
          tree_parents.insert(tree_parents.begin(),parents_first,parents_last);
        
          vector<SWC_TYPE_T>::const_iterator swc_types_first = all_swc_types.begin() + attr_start;
          vector<SWC_TYPE_T>::const_iterator swc_types_last  = all_swc_types.begin() + attr_start + attr_block;
          vector<SWC_TYPE_T> tree_swc_types;
          tree_swc_types.insert(tree_swc_types.begin(),swc_types_first,swc_types_last);

          CELL_IDX_T gid = pop_start+all_gid_vector[i];
          size_t dst_rank;
          auto it = node_rank_map.find(gid);
          if (it == node_rank_map.end())
            {
              printf("gid %d not found in node rank map\n", gid);
            }
          assert(it != node_rank_map.end());
          dst_rank = it->second;
          neurotree_t tree = make_tuple(gid,tree_src_vector,tree_dst_vector,tree_sections,
                                        tree_xcoords,tree_ycoords,tree_zcoords,
                                        tree_radiuses,tree_layers,tree_parents,
                                        tree_swc_types);
          map<CELL_IDX_T, neurotree_t> &tree_map = rank_tree_map[dst_rank];
          tree_map.insert(make_pair(gid, tree));
                                   
        }
    }
  }
}
