// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attrval.hh
///
///  Functions for storing attributes in vectors of different types.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef ATTR_VAL_HH
#define ATTR_VAL_HH

#include <map>
#include <vector>

#include "neurotrees_types.h"

namespace neurotrees
{

    struct AttrVal
    {
      std::vector < std::vector <float> > float_values;
      std::vector < std::vector <uint8_t> > uint8_values;
      std::vector < std::vector <uint16_t> > uint16_values;
      std::vector < std::vector <uint32_t> > uint32_values;

      template<class T>
      const size_t size_attr_vec () const;

      template<class T>
      const void resize (size_t size);
      
      template<class T>
      const std::vector<T>& attr_vec (size_t i) const; 

      template<class T>
      size_t size_attr (size_t i) const;

      size_t insert (const std::vector<CELL_IDX_T> &gid,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<float> &value)
      {
        size_t index;
        index = float_values.size();
        float_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<uint8_t> &value)
      {
        size_t index;
        index = uint8_values.size();
        uint8_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<uint16_t> &value)
      {
        size_t index;
        index = uint16_values.size();
        uint16_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<uint32_t> &value)
      {
        size_t index;
        index = uint32_values.size();
        uint32_values.push_back(value);
        return index;
      }

      template<class T>
      void push_back (size_t vindex, T value);

      template<class T>
      const T at (size_t vindex, size_t index) const;

      void append (AttrVal a)
      {
        float_values.insert(float_values.end(),a.float_values.begin(),
                            a.float_values.end());
        uint8_values.insert(uint8_values.end(),a.uint8_values.begin(),
                            a.uint8_values.end());
        uint16_values.insert(uint16_values.end(),a.uint16_values.begin(),
                             a.uint16_values.end());
        uint32_values.insert(uint32_values.end(),a.uint32_values.begin(),
                             a.uint32_values.end());
      }
    };

    

    struct NamedAttrVal : AttrVal
    {

      std::map<std::string, size_t> float_names;
      std::map<std::string, size_t> uint8_names;
      std::map<std::string, size_t> uint16_names;
      std::map<std::string, size_t> uint32_names;

      size_t insert (std::string name, const std::vector<float> &value)
      {
        size_t index;
        index = float_values.size();
        float_values.push_back(value);
        float_names.insert(make_pair(name, index));
        return index;
      }

      size_t insert (std::string name, const std::vector<uint8_t> &value)
      {
        size_t index;
        index = uint8_values.size();
        uint8_values.push_back(value);
        uint8_names.insert(make_pair(name, index));
        return index;
      }

      size_t insert (std::string name, const std::vector<uint16_t> &value)
      {
        size_t index;
        index = uint16_values.size();
        uint16_values.push_back(value);
        uint16_names.insert(make_pair(name, index));
        return index;
      }

      size_t insert (std::string name, const std::vector<uint32_t> &value)
      {
        size_t index;
        index = uint32_values.size();
        uint32_values.push_back(value);
        uint32_names.insert(make_pair(name, index));
        return index;
      }
    };
}

#endif
