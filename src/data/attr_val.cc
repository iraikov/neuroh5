// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attrval.cc
///
///  Template specialization for AttrVal. 
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "attrval.hh"

#include <map>
#include <vector>

using namespace std;
using namespace neuroio;

namespace neuroio
{

    template<>
    const size_t AttrVal::size_attr_vec<float> () const
    {
      return this->float_values.size();
    }
    template<>
    const size_t AttrVal::size_attr_vec<uint8_t> () const
    {
      return this->uint8_values.size();
    }
    template<>
    const size_t AttrVal::size_attr_vec<uint16_t> () const
    {
      return this->uint16_values.size();
    }
    template<>
    const size_t AttrVal::size_attr_vec<uint32_t> () const
    {
      return this->uint32_values.size();
    }

    template<>
    const void AttrVal::resize<float> (size_t size)
    {
      float_values.resize(size);
    }

    template<>
    const void AttrVal::resize<uint8_t> (size_t size)
    {
      uint8_values.resize(size);
    }

    template<>
    const void AttrVal::resize<uint16_t> (size_t size)
    {
      uint16_values.resize(size);
    }

    template<>
    const void AttrVal::resize<uint32_t> (size_t size)
    {
      uint32_values.resize(size);
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
    size_t AttrVal::size_attr<uint16_t> (size_t i) const
      {
        return uint16_values[i].size();
      }

    template<>
    size_t AttrVal::size_attr<uint32_t> (size_t i) const
      {
        return uint32_values[i].size();
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
    void AttrVal::push_back<uint16_t> (size_t vindex, uint16_t value)
      {
        uint16_values[vindex].push_back(value);
      }

    template<>
    void AttrVal::push_back<uint32_t> (size_t vindex, uint32_t value)
      {
        uint32_values[vindex].push_back(value);
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
    const uint16_t AttrVal::at<uint16_t> (size_t vindex, size_t index) const
      {
        return uint16_values[vindex][index];
      }

    template<>
    const uint32_t AttrVal::at<uint32_t> (size_t vindex, size_t index) const
      {
        return uint32_values[vindex][index];
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
    const std::vector<uint16_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->uint16_values[i];
    }

    template<>
    const std::vector<uint32_t>& AttrVal::attr_vec (size_t i)  const
    {
      return this->uint32_values[i];
    }

}
