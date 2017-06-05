// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_val.hh
///
///  Functions for storing attributes in vectors of different types.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef ATTR_VAL_HH
#define ATTR_VAL_HH

#include <map>
#include <vector>
#include <cassert>

namespace neuroh5
{
  namespace data
  {

    struct AttrVal
    {
      static const size_t num_attr_types = 4;
      static const size_t attr_index_float  = 0;
      static const size_t attr_index_uint8  = 1;
      static const size_t attr_index_uint16 = 2;
      static const size_t attr_index_uint32 = 3;

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

      size_t insert (const std::vector<float> &value)
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
        assert(float_values.size() == a.float_values.size());
        assert(uint8_values.size() == a.uint8_values.size());
        assert(uint16_values.size() == a.uint16_values.size());
        assert(uint32_values.size() == a.uint32_values.size());
        for (size_t i=0; i<float_values.size(); i++)
          {
            float_values[i].insert(float_values[i].end(),
                                   a.float_values[i].begin(),
                                   a.float_values[i].end());
          }
        for (size_t i=0; i<uint8_values.size(); i++)
          {
            uint8_values[i].insert(uint8_values[i].end(),
                                   a.uint8_values[i].begin(),
                                   a.uint8_values[i].end());
          }
        for (size_t i=0; i<uint16_values.size(); i++)
          {
            uint16_values[i].insert(uint16_values[i].end(),
                                   a.uint16_values[i].begin(),
                                   a.uint16_values[i].end());
          }
        for (size_t i=0; i<uint32_values.size(); i++)
          {
            uint32_values[i].insert(uint32_values[i].end(),
                                   a.uint32_values[i].begin(),
                                   a.uint32_values[i].end());
          }

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

      void attr_names (std::vector<std::vector<std::string> > &) const; 

    };
  }
}

#endif
