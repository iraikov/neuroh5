
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_edge_map.cc
///
///  Populates a mapping between ranks and attribute values.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_val.hh"
#include "attr_map.hh"
#include "rank_range.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {

    void append_rank_attr_map
    (
     const data::NamedAttrMap   &attr_values,
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     map <rank_t, data::AttrMap> &rank_attr_map)
    {
      const vector<map< CELL_IDX_T, vector<float> > > &all_float_values     = attr_values.attr_maps<float>();
      const vector<map< CELL_IDX_T, vector<int8_t> > > &all_int8_values     = attr_values.attr_maps<int8_t>();
      const vector<map< CELL_IDX_T, vector<uint8_t> > > &all_uint8_values   = attr_values.attr_maps<uint8_t>();
      const vector<map< CELL_IDX_T, vector<uint16_t> > > &all_uint16_values = attr_values.attr_maps<uint16_t>();
      const vector<map< CELL_IDX_T, vector<int16_t> > > &all_int16_values   = attr_values.attr_maps<int16_t>();
      const vector<map< CELL_IDX_T, vector<uint32_t> > > &all_uint32_values = attr_values.attr_maps<uint32_t>();
      const vector<map< CELL_IDX_T, vector<int32_t> > > &all_int32_values   = attr_values.attr_maps<int32_t>();
    
      for (size_t i=0; i<all_float_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<float> > &float_values = all_float_values[i];
          for (auto const& element : float_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<float> &v = element.second;
              auto it = node_rank_map.find(index);
              if(it == node_rank_map.end())
                {
                  printf("index %u not in node rank map\n", index);
                }
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint8_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint8_t> > &uint8_values = all_uint8_values[i];
          for (auto const& element : uint8_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int8_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int8_t> > &int8_values = all_int8_values[i];
          for (auto const& element : int8_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint16_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint16_t> > &uint16_values = all_uint16_values[i];
          for (auto const& element : uint16_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int16_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int16_t> > &int16_values = all_int16_values[i];
          for (auto const& element : int16_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint32_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint32_t> > &uint32_values = all_uint32_values[i];
          for (auto const& element : uint32_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int32_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int32_t> > &int32_values = all_int32_values[i];
          for (auto const& element : int32_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert(it != node_rank_map.end(),
                           "append_rank_attr_map: index not found in node rank map");
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

    }
  }
}
