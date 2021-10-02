// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.cc
///
///  Template specialization for AttrMap. 
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include <map>
#include <set>
#include <vector>
#include <deque>

#include "attr_map.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {


    
  template<>
  const map< CELL_IDX_T, deque<float> >& AttrMap::attr_map<float> (size_t i) const
  {
    return float_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<uint8_t> >& AttrMap::attr_map<uint8_t> (size_t i) const
  {
    return uint8_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<int8_t> >& AttrMap::attr_map<int8_t> (size_t i) const
  {
    return int8_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<int16_t> >& AttrMap::attr_map<int16_t> (size_t i) const
  {
    return int16_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<uint16_t> >& AttrMap::attr_map<uint16_t> (size_t i) const
  {
    return uint16_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<uint32_t> >& AttrMap::attr_map<uint32_t> (size_t i) const
  {
    return uint32_values[i];
  }
  template<>
  const map< CELL_IDX_T, deque<int32_t> >& AttrMap::attr_map<int32_t> (size_t i) const
  {
    return int32_values[i];
  }
    
  template<>
  map< CELL_IDX_T, deque<float> >& AttrMap::attr_map<float> (size_t i)
  {
    return float_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<uint8_t> >& AttrMap::attr_map<uint8_t> (size_t i)
  {
    return uint8_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<int8_t> >& AttrMap::attr_map<int8_t> (size_t i)
  {
    return int8_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<int16_t> >& AttrMap::attr_map<int16_t> (size_t i)
  {
    return int16_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<uint16_t> >& AttrMap::attr_map<uint16_t> (size_t i)
  {
    return uint16_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<uint32_t> >& AttrMap::attr_map<uint32_t> (size_t i)
  {
    return uint32_values[i];
  }
  template<>
  map< CELL_IDX_T, deque<int32_t> >& AttrMap::attr_map<int32_t> (size_t i)
  {
    return int32_values[i];
  }

  template<>
  const vector<map< CELL_IDX_T, deque<float> > >& AttrMap::attr_maps<float> () const
  {
    return float_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<uint8_t> > >& AttrMap::attr_maps<uint8_t> () const
  {
    return uint8_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<int8_t> > >& AttrMap::attr_maps<int8_t> () const
  {
    return int8_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<uint16_t> > >& AttrMap::attr_maps<uint16_t> () const
  {
    return uint16_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<int16_t> > >& AttrMap::attr_maps<int16_t> () const
  {
    return int16_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<uint32_t> > >& AttrMap::attr_maps<uint32_t> () const
  {
    return uint32_values;
  }
  template<>
  const vector<map< CELL_IDX_T, deque<int32_t> > >& AttrMap::attr_maps<int32_t> () const
  {
    return int32_values;
  }

  template<>
  vector<map< CELL_IDX_T, deque<float> > >& AttrMap::attr_maps<float> ()
  {
    return float_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<uint8_t> > >& AttrMap::attr_maps<uint8_t> ()
  {
    return uint8_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<int8_t> > >& AttrMap::attr_maps<int8_t> ()
  {
    return int8_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<uint16_t> > >& AttrMap::attr_maps<uint16_t> ()
  {
    return uint16_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<int16_t> > >& AttrMap::attr_maps<int16_t> ()
  {
    return int16_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<uint32_t> > >& AttrMap::attr_maps<uint32_t> ()
  {
    return uint32_values;
  }
  template<>
  vector<map< CELL_IDX_T, deque<int32_t> > >& AttrMap::attr_maps<int32_t> ()
  {
    return int32_values;
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
  

  void NamedAttrMap::attr_names (vector<vector<string>> &attr_names) const
  {
    attr_names.resize(AttrMap::num_attr_types);
    attr_names_type<float>(attr_names[AttrMap::attr_index_float]);
    attr_names_type<int8_t>(attr_names[AttrMap::attr_index_int8]);
    attr_names_type<int16_t>(attr_names[AttrMap::attr_index_int16]);
    attr_names_type<int32_t>(attr_names[AttrMap::attr_index_int32]);
    attr_names_type<uint8_t>(attr_names[AttrMap::attr_index_uint8]);
    attr_names_type<uint16_t>(attr_names[AttrMap::attr_index_uint16]);
    attr_names_type<uint32_t>(attr_names[AttrMap::attr_index_uint32]);
  }
  

    
  }
}
