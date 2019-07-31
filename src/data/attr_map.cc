// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.cc
///
///  Template specialization for AttrMap. 
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include <map>
#include <set>
#include <vector>

#include "attr_map.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {


    
  template<>
  const map< CELL_IDX_T, vector<float> >& AttrMap::attr_map<float> (size_t i) const
  {
    return float_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<uint8_t> >& AttrMap::attr_map<uint8_t> (size_t i) const
  {
    return uint8_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<int8_t> >& AttrMap::attr_map<int8_t> (size_t i) const
  {
    return int8_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<int16_t> >& AttrMap::attr_map<int16_t> (size_t i) const
  {
    return int16_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<uint16_t> >& AttrMap::attr_map<uint16_t> (size_t i) const
  {
    return uint16_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<uint32_t> >& AttrMap::attr_map<uint32_t> (size_t i) const
  {
    return uint32_values[i];
  }
  template<>
  const map< CELL_IDX_T, vector<int32_t> >& AttrMap::attr_map<int32_t> (size_t i) const
  {
    return int32_values[i];
  }

  template<>
  const vector<map< CELL_IDX_T, vector<float> > >& AttrMap::attr_maps<float> () const
  {
    return float_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<uint8_t> > >& AttrMap::attr_maps<uint8_t> () const
  {
    return uint8_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<int8_t> > >& AttrMap::attr_maps<int8_t> () const
  {
    return int8_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<uint16_t> > >& AttrMap::attr_maps<uint16_t> () const
  {
    return uint16_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<int16_t> > >& AttrMap::attr_maps<int16_t> () const
  {
    return int16_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<uint32_t> > >& AttrMap::attr_maps<uint32_t> () const
  {
    return uint32_values;
  }
  template<>
  const vector<map< CELL_IDX_T, vector<int32_t> > >& AttrMap::attr_maps<int32_t> () const
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
  size_t AttrMap::num_attr<int16_t> () const
  {
    return int16_values.size();
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
    v[AttrMap::attr_index_int16]=num_attr<int16_t>();
    v[AttrMap::attr_index_uint32]=num_attr<uint32_t>();
    v[AttrMap::attr_index_int32]=num_attr<int32_t>();
  }

  template<>
  const vector<vector<float>> AttrMap::find<float> (CELL_IDX_T index) const
  {
    vector< vector<float> > result;
    for (size_t i =0; i<float_values.size(); i++)
      {
        float_values_const_iter it = float_values[i].find(index);
        if (it != float_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  
  template<>
  const vector<vector<uint8_t>> AttrMap::find<uint8_t> (CELL_IDX_T index) const
  {
    vector< vector<uint8_t> > result;
    for (size_t i =0; i<uint8_values.size(); i++)
      {
        uint8_values_const_iter it = uint8_values[i].find(index);
        if (it != uint8_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  template<>
  const vector<vector<int8_t>> AttrMap::find<int8_t> (CELL_IDX_T index) const
  {
    
    vector< vector<int8_t> > result;
    for (size_t i =0; i<int8_values.size(); i++)
      {
        int8_values_const_iter it = int8_values[i].find(index);
        if (it != int8_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  template<>
  const vector<vector<uint16_t>> AttrMap::find<uint16_t> (CELL_IDX_T index) const
  {
    vector< vector<uint16_t> > result;
    for (size_t i =0; i<uint16_values.size(); i++)
      {
        uint16_values_const_iter it = uint16_values[i].find(index);
        if (it != uint16_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  template<>
  const vector<vector<int16_t>> AttrMap::find<int16_t> (CELL_IDX_T index) const
  {
    vector< vector<int16_t> > result;
    for (size_t i =0; i<int16_values.size(); i++)
      {
        int16_values_const_iter it = int16_values[i].find(index);
        if (it != int16_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  template<>
  const vector<vector<uint32_t>> AttrMap::find<uint32_t> (CELL_IDX_T index) const
  {
    vector< vector<uint32_t> > result;
    for (size_t i =0; i<uint32_values.size(); i++)
      {
        uint32_values_const_iter it = uint32_values[i].find(index);
        if (it != uint32_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }
  template<>
  const vector<vector<int32_t>> AttrMap::find<int32_t> (CELL_IDX_T index) const
  {
    vector< vector<int32_t> > result;
    for (size_t i =0; i<int32_values.size(); i++)
      {
        int32_values_const_iter it = int32_values[i].find(index);
        if (it != int32_values[i].end())
          {
            result.push_back(it->second);
          }
      }
    return result;
  }

  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<float> &value)
  {
    size_t index = float_values.size();
    float_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<float>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(float_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<uint8_t> &value)
  {
    size_t index = uint8_values.size();
    uint8_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<uint8_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(uint8_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<int8_t> &value)
  {
    size_t index = int8_values.size();
    int8_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<int8_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(int8_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }

  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<uint16_t> &value)
  {
    size_t index = uint16_values.size();
    uint16_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<uint16_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(uint16_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<int16_t> &value)
  {
    size_t index = int16_values.size();
    int16_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<int16_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(int16_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<uint32_t> &value)
  {
    size_t index = uint32_values.size();
    uint32_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<uint32_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(uint32_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const std::vector<CELL_IDX_T> &cell_index,
                          const std::vector<ATTR_PTR_T> &ptr,
                          const std::vector<int32_t> &value)
  {
    size_t index = int32_values.size();
    int32_values.resize(index+1);
    for (size_t p=0; p<cell_index.size(); p++)
      {
        CELL_IDX_T vindex = cell_index[p];
        vector<int32_t>::const_iterator first, last;
        if (ptr.size() > 1)
          {
            first = value.begin() + ptr[p];
            last  = value.begin() + ptr[p+1];
          }
        else
          {
            first = value.begin();
            last  = value.end();
          }
        append_values_map(int32_values[index], vindex, first, last);
        index_set.insert(vindex);
      }
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<float> &value)
  {
    float_values.resize(max(float_values.size(), index+1));
    float_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }
  
  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<uint8_t> &value)
  {
    uint8_values.resize(max(uint8_values.size(), index+1));
    uint8_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<int8_t> &value)
  {
    int8_values.resize(max(int8_values.size(), index+1));
    int8_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<uint16_t> &value)
  {
    uint16_values.resize(max(uint16_values.size(), index+1));
    uint16_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }
  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<int16_t> &value)
  {
    int16_values.resize(max(int16_values.size(), index+1));
    int16_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }
  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<uint32_t> &value)
  {
    uint32_values.resize(max(uint32_values.size(), index+1));
    uint32_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t AttrMap::insert (const size_t index,
                          const CELL_IDX_T &cell_index,
                          const std::vector<int32_t> &value)
  {
    int32_values.resize(max(int32_values.size(), index+1));
    int32_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
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
  const map<string, size_t>& NamedAttrMap::attr_name_map<int16_t> () const
  {
    return int16_names;
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
    const map< string, size_t> &int16_attr_names  = attr_name_map<int16_t>();
    const map< string, size_t> &uint16_attr_names = attr_name_map<uint16_t>();
    const map< string, size_t> &uint32_attr_names = attr_name_map<uint32_t>();
    const map< string, size_t> &int32_attr_names  = attr_name_map<int32_t>();

    attr_names.resize(AttrMap::num_attr_types);
    attr_names[AttrMap::attr_index_float].resize(float_attr_names.size());
    attr_names[AttrMap::attr_index_uint8].resize(uint8_attr_names.size());
    attr_names[AttrMap::attr_index_int8].resize(int8_attr_names.size());
    attr_names[AttrMap::attr_index_uint16].resize(uint16_attr_names.size());
    attr_names[AttrMap::attr_index_int16].resize(int16_attr_names.size());
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
    for (auto const& element : int16_attr_names)
      {
        attr_names[AttrMap::attr_index_int16][element.second] = string(element.first);
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
    float_values.resize(max(float_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<uint8_t> (string name, size_t index)
  {
    uint8_names.insert(make_pair(name, index));
    uint8_values.resize(max(uint8_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<int8_t> (string name, size_t index)
  {
    int8_names.insert(make_pair(name, index));
    int8_values.resize(max(int8_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<uint16_t> (string name, size_t index)
  {
    uint16_names.insert(make_pair(name, index));
    uint16_values.resize(max(uint16_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<int16_t> (string name, size_t index)
  {
    int16_names.insert(make_pair(name, index));
    int16_values.resize(max(int16_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<uint32_t> (string name, size_t index)
  {
    uint32_names.insert(make_pair(name, index));
    uint32_values.resize(max(uint32_values.size(), index+1));
  }
  template<>
  void NamedAttrMap::insert_name<int32_t> (string name, size_t index)
  {
    int32_names.insert(make_pair(name, index));
    int32_values.resize(max(int32_values.size(), index+1));
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<float> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<float>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<uint8_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<uint8_t>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<int8_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<int8_t>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<uint16_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<uint16_t>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<int16_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<int16_t>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<uint32_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<uint32_t>(name, index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (std::string name,
                               const std::vector<CELL_IDX_T> &cell_index,
                               const std::vector<ATTR_PTR_T> &ptr,
                               const std::vector<int32_t> &value)
  {
    size_t index = AttrMap::insert(cell_index, ptr, value);
    insert_name<int32_t>(name, index);
    return index;
  }


  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<float> &value)
  {
    if (float_values.size() <= index)
      {
        float_values.resize(index+1);
      }
    float_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<uint8_t> &value)
  {
    if (uint8_values.size() <= index)
      {
        uint8_values.resize(index+1);
      }
    uint8_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<int8_t> &value)
  {
    if (int8_values.size() <= index)
      {
        int8_values.resize(index+1);
      }
    int8_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }
  
  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<uint16_t> &value)
  {
    if (uint16_values.size() <= index)
      {
        uint16_values.resize(index+1);
      }
    uint16_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<int16_t> &value)
  {
    if (int16_values.size() <= index)
      {
        int16_values.resize(index+1);
      }
    int16_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<uint32_t> &value)
  {
    if (uint32_values.size() <= index)
      {
        uint32_values.resize(index+1);
      }
    uint32_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

  template<>
  size_t NamedAttrMap::insert (const size_t index,
                               const CELL_IDX_T &cell_index,
                               const std::vector<int32_t> &value)
  {
    if (int32_values.size() <= index)
      {
        int32_values.resize(index+1);
      }
    int32_values[index].insert(make_pair(cell_index, value));
    index_set.insert(cell_index);
    return index;
  }

    
  template<>
  const vector<float> NamedAttrMap::find_name<float> (const std::string& name,
                                                      CELL_IDX_T& index)
  {
    vector<float> result;
    auto attr_it = float_names.find(name);
    throw_assert(attr_it != float_names.end(),
                 "NamedAttrMap::find_name<float>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = float_values[attr_index].find(index);
    if (it != float_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }

  template<>
  const vector<uint8_t> NamedAttrMap::find_name<uint8_t> (const std::string& name,
                                                          CELL_IDX_T& index)
  {
    vector<uint8_t> result;
    auto attr_it = uint8_names.find(name);
    throw_assert(attr_it != uint8_names.end(),
                 "NamedAttrMap::find_name<uint8_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = uint8_values[attr_index].find(index);
    if (it != uint8_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }

  template<>
  const vector<int8_t> NamedAttrMap::find_name<int8_t> (const std::string& name,
                                                        CELL_IDX_T& index)
  {
    vector<int8_t> result;
    auto attr_it = int8_names.find(name);
    throw_assert(attr_it != int8_names.end(),
                 "NamedAttrMap::find_name<int8_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = int8_values[attr_index].find(index);
    if (it != int8_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }


  template<>
  const vector<uint16_t> NamedAttrMap::find_name<uint16_t> (const std::string& name,
                                                            CELL_IDX_T& index)
  {
    vector<uint16_t> result;
    auto attr_it = uint16_names.find(name);
    throw_assert(attr_it != int16_names.end(),
                 "NamedAttrMap::find_name<uint16_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = uint16_values[attr_index].find(index);
    if (it != uint16_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }

  template<>
  const vector<int16_t> NamedAttrMap::find_name<int16_t> (const std::string& name,
                                                      CELL_IDX_T& index)
  {
    vector<int16_t> result;
    auto attr_it = int16_names.find(name);
    throw_assert(attr_it != int16_names.end(),
                 "NamedAttrMap::find_name<int16_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = int16_values[attr_index].find(index);
    if (it != int16_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }


  template<>
  const vector<uint32_t> NamedAttrMap::find_name<uint32_t> (const std::string& name,
                                                            CELL_IDX_T& index)
  {
    vector<uint32_t> result;
    auto attr_it = uint32_names.find(name);
    throw_assert(attr_it != uint32_names.end(),
                 "NamedAttrMap::find_name<uint32_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = uint32_values[attr_index].find(index);
    if (it != uint32_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }

  template<>
  const vector<int32_t> NamedAttrMap::find_name<int32_t> (const std::string& name,
                                                          CELL_IDX_T& index)
  {
    vector<int32_t> result;
    auto attr_it = int32_names.find(name);
    throw_assert(attr_it != int32_names.end(),
                 "NamedAttrMap::find_name<int32_t>: attribute " << name << " not found");
    size_t attr_index = attr_it->second;
    auto it = int32_values[attr_index].find(index);
    if (it != int32_values[attr_index].end())
      {
        result = it->second;
      }
    return result;
  }

    
  }
}
