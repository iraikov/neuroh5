
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_tree_map.cc
///
///  Populates a mapping between ranks and tree values.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include <deque>
#include <vector>
#include <map>
#include <set>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {

    void append_rank_tree_map (NamedAttrMap& attr_values,
                               const node_rank_map_t& node_rank_map,
                               map <rank_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map)
    {
      for (CELL_IDX_T gid : attr_values.index_set)
        {
          const deque<SECTION_IDX_T>& src_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::SRCSEC, gid);
          const deque<SECTION_IDX_T>& dst_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::DSTSEC, gid);

          const deque<SECTION_IDX_T>& sections = attr_values.find_name<SECTION_IDX_T>(hdf5::SECTION, gid);
        
          const deque<COORD_T>& xcoords =  attr_values.find_name<COORD_T>(hdf5::X_COORD, gid);
          const deque<COORD_T>& ycoords =  attr_values.find_name<COORD_T>(hdf5::Y_COORD, gid);
          const deque<COORD_T>& zcoords =  attr_values.find_name<COORD_T>(hdf5::Z_COORD, gid);

          const deque<REALVAL_T>& radiuses    =  attr_values.find_name<REALVAL_T>(hdf5::RADIUS, gid);
          const deque<LAYER_IDX_T>& layers    =  attr_values.find_name<LAYER_IDX_T>(hdf5::LAYER, gid);
          const deque<PARENT_NODE_IDX_T>& parents = attr_values.find_name<PARENT_NODE_IDX_T>(hdf5::PARENT, gid);
          const deque<SWC_TYPE_T> swc_types   = attr_values.find_name<SWC_TYPE_T>(hdf5::SWCTYPE, gid);

          set <rank_t> dst_rank_set;
          auto it = node_rank_map.find(gid);
          if (it == node_rank_map.end())
            {
              printf("gid %d not found in node rank map\n", gid);
            }
          throw_assert(it != node_rank_map.end(),
                       "append_rank_tree_map: index not found in node rank map");

          dst_rank_set = it->second;
          
          neurotree_t tree = make_tuple(gid, src_vector, dst_vector, sections,
                                        xcoords, ycoords, zcoords,
                                        radiuses, layers, parents,
                                        swc_types);

          for (auto dst_rank : dst_rank_set)
            {
              map<CELL_IDX_T, neurotree_t> &tree_map = rank_tree_map[dst_rank];
              tree_map.insert(make_pair(gid, tree));
            }
                                   
        }
    }
  }
}
