// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_val.cc
///
///  Template specialization for AttrVal. 
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include <map>
#include <vector>

#include "attr_val.hh"

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace data
  {

    template<>
    size_t AttrVal::attr_type_index<float> () 
    {
      return AttrVal::attr_index_float;
    }
    template<>
    size_t AttrVal::attr_type_index<uint8_t> ()
    {
      return AttrVal::attr_index_uint8;
    }
    template<>
    size_t AttrVal::attr_type_index<uint16_t> () 
    {
      return AttrVal::attr_index_uint16;
    }
    template<>
    size_t AttrVal::attr_type_index<uint32_t> () 
    {
      return AttrVal::attr_index_uint32;
    }
    template<>
    size_t AttrVal::attr_type_index<int8_t> () 
    {
      return AttrVal::attr_index_int8;
    }
    template<>
    size_t AttrVal::attr_type_index<int16_t> ()
    {
      return AttrVal::attr_index_int16;
    }
    template<>
    size_t AttrVal::attr_type_index<int32_t> ()
    {
      return AttrVal::attr_index_int32;
    }
    
    template<>
    size_t AttrVal::size_attr_vec<float> () const
    {
      return this->float_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<uint8_t> () const
    {
      return this->uint8_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<int8_t> () const
    {
      return this->int8_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<uint16_t> () const
    {
      return this->uint16_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<int16_t> () const
    {
      return this->int16_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<uint32_t> () const
    {
      return this->uint32_values.size();
    }
    template<>
    size_t AttrVal::size_attr_vec<int32_t> () const
    {
      return this->int32_values.size();
    }

    template<>
    void AttrVal::resize<float> (size_t size)
    {
      float_values.resize(size);
    }

    template<>
    void AttrVal::resize<uint8_t> (size_t size)
    {
      uint8_values.resize(size);
    }
    template<>
    void AttrVal::resize<int8_t> (size_t size)
    {
      int8_values.resize(size);
    }

    template<>
    void AttrVal::resize<uint16_t> (size_t size)
    {
      uint16_values.resize(size);
    }
    template<>
    void AttrVal::resize<int16_t> (size_t size)
    {
      int16_values.resize(size);
    }

    template<>
    void AttrVal::resize<uint32_t> (size_t size)
    {
      uint32_values.resize(size);
    }
    template<>
    void AttrVal::resize<int32_t> (size_t size)
    {
      int32_values.resize(size);
    }

    template<>
    size_t AttrVal::size_attr<float> (size_t i) const
      {
        return float_values[i].size();
      }

    template<>
    size_t AttrVal::size_attr<uint8_t> (size_t i) const
      {
        return uint8_values[i].size();
      }

    template<>
    size_t AttrVal::size_attr<int8_t> (size_t i) const
      {
        return int8_values[i].size();
      }

    template<>
    size_t AttrVal::size_attr<uint16_t> (size_t i) const
      {
        return uint16_values[i].size();
      }
    template<>
    size_t AttrVal::size_attr<int16_t> (size_t i) const
      {
        return int16_values[i].size();
      }

    template<>
    size_t AttrVal::size_attr<uint32_t> (size_t i) const
      {
        return uint32_values[i].size();
      }
    template<>
    size_t AttrVal::size_attr<int32_t> (size_t i) const
      {
        return int32_values[i].size();
      }

    template<>
    void AttrVal::push_back<float> (size_t vindex, float value)
      {
        float_values[vindex].push_back(value);
      }

    template<>
    void AttrVal::push_back<uint8_t> (size_t vindex, uint8_t value)
      {
        uint8_values[vindex].push_back(value);
      }
    template<>
    void AttrVal::push_back<int8_t> (size_t vindex, int8_t value)
      {
        int8_values[vindex].push_back(value);
      }

    template<>
    void AttrVal::push_back<uint16_t> (size_t vindex, uint16_t value)
      {
        uint16_values[vindex].push_back(value);
      }
    template<>
    void AttrVal::push_back<int16_t> (size_t vindex, int16_t value)
      {
        int16_values[vindex].push_back(value);
      }

    template<>
    void AttrVal::push_back<uint32_t> (size_t vindex, uint32_t value)
      {
        uint32_values[vindex].push_back(value);
      }
    template<>
    void AttrVal::push_back<int32_t> (size_t vindex, int32_t value)
      {
        int32_values[vindex].push_back(value);
      }

    template<>
    const float AttrVal::at<float> (size_t vindex, size_t index) const
      {
        return float_values[vindex][index];
      }

    template<>
    const uint8_t AttrVal::at<uint8_t> (size_t vindex, size_t index) const
      {
        return uint8_values[vindex][index];
      }
    template<>
    const int8_t AttrVal::at<int8_t> (size_t vindex, size_t index) const
      {
        return uint8_values[vindex][index];
      }

    template<>
    const uint16_t AttrVal::at<uint16_t> (size_t vindex, size_t index) const
      {
        return uint16_values[vindex][index];
      }
    template<>
    const int16_t AttrVal::at<int16_t> (size_t vindex, size_t index) const
      {
        return int16_values[vindex][index];
      }

    template<>
    const uint32_t AttrVal::at<uint32_t> (size_t vindex, size_t index) const
      {
        return uint32_values[vindex][index];
      }
    template<>
    const int32_t AttrVal::at<int32_t> (size_t vindex, size_t index) const
      {
        return int32_values[vindex][index];
      }
    
    template<>
    const std::vector<float>& AttrVal::attr_vec (size_t i)  const
    {
      return this->float_values[i];
    }
    
    template<>
    const std::vector<uint8_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->uint8_values[i];
    }
    template<>
    const std::vector<int8_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->int8_values[i];
    }
    
    template<>
    const std::vector<uint16_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->uint16_values[i];
    }
    template<>
    const std::vector<int16_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->int16_values[i];
    }

    template<>
    const std::vector<uint32_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->uint32_values[i];
    }
    template<>
    const std::vector<int32_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->int32_values[i];
    }

    void NamedAttrVal::attr_names (vector<vector<string>> &attr_names) const
    {
      attr_names.resize(NamedAttrVal::num_attr_types);
      attr_names[NamedAttrVal::attr_index_float].resize(float_names.size());
      attr_names[NamedAttrVal::attr_index_uint8].resize(uint8_names.size());
      attr_names[NamedAttrVal::attr_index_int8].resize(int8_names.size());
      attr_names[NamedAttrVal::attr_index_uint16].resize(uint16_names.size());
      attr_names[NamedAttrVal::attr_index_int16].resize(int16_names.size());
      attr_names[NamedAttrVal::attr_index_uint32].resize(uint32_names.size());
      attr_names[NamedAttrVal::attr_index_int32].resize(int32_names.size());
        
      for (auto const& element : float_names)
        {
          attr_names[NamedAttrVal::attr_index_float][element.second] = string(element.first);
        }
      for (auto const& element : uint8_names)
        {
          attr_names[NamedAttrVal::attr_index_uint8][element.second] = string(element.first);
        }
      for (auto const& element : int8_names)
        {
          attr_names[NamedAttrVal::attr_index_int8][element.second] = string(element.first);
        }
      for (auto const& element : uint16_names)
        {
          attr_names[NamedAttrVal::attr_index_uint16][element.second] = string(element.first);
        }
      for (auto const& element : int16_names)
        {
          attr_names[NamedAttrVal::attr_index_int16][element.second] = string(element.first);
        }
      for (auto const& element : uint32_names)
        {
          attr_names[NamedAttrVal::attr_index_uint32][element.second] = string(element.first);
        }
      for (auto const& element : int32_names)
        {
          attr_names[NamedAttrVal::attr_index_int32][element.second] = string(element.first);
        }
 
    }

    
  }
}
