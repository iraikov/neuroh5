
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_tree_map.cc
///
///  Populates a mapping between ranks and tree values.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include <vector>
#include <map>

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

    void append_tree_map (NamedAttrMap& attr_values,
			  map<CELL_IDX_T, neurotree_t> &tree_map)
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

          neurotree_t tree = make_tuple(gid,src_vector,dst_vector,sections,
                                        xcoords,ycoords,zcoords,
                                        radiuses,layers,parents,
                                        swc_types);
          
          tree_map.insert(make_pair(gid, tree));
                                   
        }
    }
  }
}
