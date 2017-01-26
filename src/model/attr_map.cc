// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.cc
///
///  Template specialization for AttrMap. 
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "attr_map.hh"

#include <map>
#include <set>
#include <vector>

using namespace std;
using namespace ngh5;

namespace ngh5
{
  namespace model
  {

    template<>
    const map< NODE_IDX_T, float >& AttrMap::attr_map<float> (size_t i) const
    {
      return float_values[i];
    }
    template<>
    const map< NODE_IDX_T, uint8_t >& AttrMap::attr_map<uint8_t> (size_t i) const
    {
      return uint8_values[i];
    }
    template<>
    const map< NODE_IDX_T, int8_t >& AttrMap::attr_map<int8_t> (size_t i) const
    {
      return int8_values[i];
    }
    template<>
    const map< NODE_IDX_T, uint16_t >& AttrMap::attr_map<uint16_t> (size_t i) const
    {
      return uint16_values[i];
    }
    template<>
    const map< NODE_IDX_T, uint32_t >& AttrMap::attr_map<uint32_t> (size_t i) const
    {
      return uint32_values[i];
    }
    template<>
    const map< NODE_IDX_T, int32_t >& AttrMap::attr_map<int32_t> (size_t i) const
    {
      return int32_values[i];
    }

    template<>
    const vector<map< NODE_IDX_T, float > >& AttrMap::attr_maps<float> () const
    {
      return float_values;
    }
    template<>
    const vector<map< NODE_IDX_T, uint8_t > >& AttrMap::attr_maps<uint8_t> () const
    {
      return uint8_values;
    }
    template<>
    const vector<map< NODE_IDX_T, int8_t > >& AttrMap::attr_maps<int8_t> () const
    {
      return int8_values;
    }
    template<>
    const vector<map< NODE_IDX_T, uint16_t > >& AttrMap::attr_maps<uint16_t> () const
    {
      return uint16_values;
    }
    template<>
    const vector<map< NODE_IDX_T, uint32_t > >& AttrMap::attr_maps<uint32_t> () const
    {
      return uint32_values;
    }
    template<>
    const vector<map< NODE_IDX_T, int32_t > >& AttrMap::attr_maps<int32_t> () const
    {
      return int32_values;
    }

  
    template<>
    size_t AttrMap::num_attr<float> () const
    {
      return float_values.size();
    }

    template<>
    size_t AttrMap::num_attr<uint8_t> () const
    {
      return uint8_values.size();
    }

    template<>
    size_t AttrMap::num_attr<int8_t> () const
    {
      return int8_values.size();
    }

    template<>
    size_t AttrMap::num_attr<uint16_t> () const
    {
      return uint16_values.size();
    }

    template<>
    size_t AttrMap::num_attr<uint32_t> () const
    {
      return uint32_values.size();
    }

    template<>
    size_t AttrMap::num_attr<int32_t> () const
    {
      return int32_values.size();
    }

    void AttrMap::num_attrs (vector<size_t> &v) const
    {
      v.resize(AttrMap::num_attr_types);
      v[AttrMap::attr_index_float]=num_attr<float>();
      v[AttrMap::attr_index_uint8]=num_attr<uint8_t>();
      v[AttrMap::attr_index_int8]=num_attr<int8_t>();
      v[AttrMap::attr_index_uint16]=num_attr<uint16_t>();
      v[AttrMap::attr_index_uint32]=num_attr<uint32_t>();
      v[AttrMap::attr_index_int32]=num_attr<int32_t>();
    }

    template<>
    const vector<float> AttrMap::find<float> (NODE_IDX_T idx)
    {
      vector<float> result;
      for (size_t i =0; i<float_values.size(); i++)
        {
          auto it = float_values[i].find(idx);
          if (it != float_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }
    template<>
    const vector<uint8_t> AttrMap::find<uint8_t> (NODE_IDX_T idx)
    {
      vector<uint8_t> result;
      for (size_t i =0; i<uint8_values.size(); i++)
        {
          auto it = uint8_values[i].find(idx);
          if (it != uint8_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }
    template<>
    const vector<int8_t> AttrMap::find<int8_t> (NODE_IDX_T idx)
    {
      vector<int8_t> result;
      for (size_t i =0; i<int8_values.size(); i++)
        {
          auto it = int8_values[i].find(idx);
          if (it != int8_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }
    template<>
    const vector<uint16_t> AttrMap::find<uint16_t> (NODE_IDX_T idx)
    {
      vector<uint16_t> result;
      for (size_t i =0; i<uint16_values.size(); i++)
        {
          auto it = uint16_values[i].find(idx);
          if (it != uint16_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }
    template<>
    const vector<uint32_t> AttrMap::find<uint32_t> (NODE_IDX_T idx)
    {
      vector<uint32_t> result;
      for (size_t i =0; i<uint32_values.size(); i++)
        {
          auto it = uint32_values[i].find(idx);
          if (it != uint32_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }
    template<>
    const vector<int32_t> AttrMap::find<int32_t> (NODE_IDX_T idx)
    {
      vector<int32_t> result;
      for (size_t i =0; i<int32_values.size(); i++)
        {
          auto it = int32_values[i].find(idx);
          if (it != int32_values[i].end())
            {
              result.push_back(it->second);
            }
        }
      return result;
    }

    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const float &value)
    {
      float_values.resize(max(float_values.size(),index+1));
      float_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }
  
    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const uint8_t &value)
    {
      uint8_values.resize(max(uint8_values.size(),index+1));
      uint8_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const int8_t &value)
    {
      int8_values.resize(max(int8_values.size(),index+1));
      int8_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const uint16_t &value)
    {
      uint16_values.resize(max(uint16_values.size(), index+1));
      uint16_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }
    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const uint32_t &value)
    {
      uint32_values.resize(max(uint32_values.size(), index+1));
      uint32_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t AttrMap::insert (const size_t index,
                            const NODE_IDX_T &idx,
                            const int32_t &value)
    {
      int32_values.resize(max(int32_values.size(), index+1));
      int32_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<float> () const
    {
      return float_names;
    }
    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<uint8_t> () const
    {
      return uint8_names;
    }
    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<int8_t> () const
    {
      return int8_names;
    }
    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<uint16_t> () const
    {
      return uint16_names;
    }
    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<uint32_t> () const
    {
      return uint32_names;
    }
    template<>
    const map<string, size_t>& NamedAttrMap::attr_name_map<int32_t> () const
    {
      return int32_names;
    }


    void NamedAttrMap::attr_names (vector<vector<string>> &attr_names) const
    {
      const map< string, size_t> &float_attr_names  = attr_name_map<float>();
      const map< string, size_t> &uint8_attr_names  = attr_name_map<uint8_t>();
      const map< string, size_t> &int8_attr_names   = attr_name_map<int8_t>();
      const map< string, size_t> &uint16_attr_names = attr_name_map<uint16_t>();
      const map< string, size_t> &uint32_attr_names = attr_name_map<uint32_t>();
      const map< string, size_t> &int32_attr_names  = attr_name_map<int32_t>();

      attr_names.resize(AttrMap::num_attr_types);
      attr_names[AttrMap::attr_index_float].resize(float_attr_names.size());
      attr_names[AttrMap::attr_index_uint8].resize(uint8_attr_names.size());
      attr_names[AttrMap::attr_index_int8].resize(int8_attr_names.size());
      attr_names[AttrMap::attr_index_uint16].resize(uint16_attr_names.size());
      attr_names[AttrMap::attr_index_uint32].resize(uint32_attr_names.size());
      attr_names[AttrMap::attr_index_int32].resize(int32_attr_names.size());
        
      for (auto const& element : float_attr_names)
        {
          attr_names[AttrMap::attr_index_float][element.second] = string(element.first);
        }
      for (auto const& element : uint8_attr_names)
        {
          attr_names[AttrMap::attr_index_uint8][element.second] = string(element.first);
        }
      for (auto const& element : int8_attr_names)
        {
          attr_names[AttrMap::attr_index_int8][element.second] = string(element.first);
        }
      for (auto const& element : uint16_attr_names)
        {
          attr_names[AttrMap::attr_index_uint16][element.second] = string(element.first);
        }
      for (auto const& element : uint32_attr_names)
        {
          attr_names[AttrMap::attr_index_uint32][element.second] = string(element.first);
        }
      for (auto const& element : int32_attr_names)
        {
          attr_names[AttrMap::attr_index_int32][element.second] = string(element.first);
        }
 
    }
  
    template<>
    void NamedAttrMap::insert_name<float> (string name, size_t index)
    {
      float_names.insert(make_pair(name, index));
    }
    template<>
    void NamedAttrMap::insert_name<uint8_t> (string name, size_t index)
    {
      uint8_names.insert(make_pair(name, index));
    }
    template<>
    void NamedAttrMap::insert_name<int8_t> (string name, size_t index)
    {
      int8_names.insert(make_pair(name, index));
    }
    template<>
    void NamedAttrMap::insert_name<uint16_t> (string name, size_t index)
    {
      uint16_names.insert(make_pair(name, index));
    }
    template<>
    void NamedAttrMap::insert_name<uint32_t> (string name, size_t index)
    {
      uint32_names.insert(make_pair(name, index));
    }
    template<>
    void NamedAttrMap::insert_name<int32_t> (string name, size_t index)
    {
      int32_names.insert(make_pair(name, index));
    }

    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const float &value)
    {
      if (float_values.size() <= index)
        {
          float_values.resize(index+1);
        }
      float_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const uint8_t &value)
    {
      if (uint8_values.size() <= index)
        {
          uint8_values.resize(index+1);
        }
      uint8_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const int8_t &value)
    {
      if (int8_values.size() <= index)
        {
          int8_values.resize(index+1);
        }
      int8_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }
  
    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const uint16_t &value)
    {
      if (uint16_values.size() <= index)
        {
          uint16_values.resize(index+1);
        }
      uint16_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const uint32_t &value)
    {
      if (uint32_values.size() <= index)
        {
          uint32_values.resize(index+1);
        }
      uint32_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

    template<>
    size_t NamedAttrMap::insert (const size_t index,
                                 const NODE_IDX_T &idx,
                                 const int32_t &value)
    {
      if (int32_values.size() <= index)
        {
          int32_values.resize(index+1);
        }
      int32_values[index].insert(make_pair(idx, value));
      idx_set.insert(idx);
      return index;
    }

  }
}
