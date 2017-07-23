
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
    (NamedAttrMap&       attr_values,
     const map<CELL_IDX_T, rank_t>& node_rank_map,
     const CELL_IDX_T pop_start,
     map <rank_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map)
    {
      for (CELL_IDX_T idx : attr_values.index_set)
        {
          const vector<SECTION_IDX_T>& src_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::SRCSEC, idx);
          const vector<SECTION_IDX_T>& dst_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::DSTSEC, idx);

          const vector<SECTION_IDX_T>& sections = attr_values.find_name<SECTION_IDX_T>(hdf5::SECTION, idx);
        
          const vector<COORD_T>& xcoords =  attr_values.find_name<COORD_T>(hdf5::X_COORD, idx);
          const vector<COORD_T>& ycoords =  attr_values.find_name<COORD_T>(hdf5::Y_COORD, idx);
          const vector<COORD_T>& zcoords =  attr_values.find_name<COORD_T>(hdf5::Z_COORD, idx);

          const vector<REALVAL_T>& radiuses    =  attr_values.find_name<REALVAL_T>(hdf5::RADIUS, idx);
          const vector<LAYER_IDX_T>& layers    =  attr_values.find_name<LAYER_IDX_T>(hdf5::LAYER, idx);
          const vector<PARENT_NODE_IDX_T>& parents = attr_values.find_name<PARENT_NODE_IDX_T>(hdf5::PARENT, idx);
          const vector<SWC_TYPE_T> swc_types   = attr_values.find_name<SWC_TYPE_T>(hdf5::SWCTYPE, idx);

          CELL_IDX_T gid = pop_start+idx;

          size_t dst_rank;
          auto it = node_rank_map.find(gid);
          if (it == node_rank_map.end())
            {
              printf("gid %d not found in node rank map\n", gid);
            }
          assert(it != node_rank_map.end());
          dst_rank = it->second;
          
          neurotree_t tree = make_tuple(gid,src_vector,dst_vector,sections,
                                        xcoords,ycoords,zcoords,
                                        radiuses,layers,parents,
                                        swc_types);
          
          map<CELL_IDX_T, neurotree_t> &tree_map = rank_tree_map[dst_rank];
          tree_map.insert(make_pair(gid, tree));
                                   
        }
    }
  }
}
