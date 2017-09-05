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

// type support
#include "cereal/types/tuple.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"


namespace neuroh5
{
  namespace data
  {

    struct AttrVal
    {
      static const size_t num_attr_types = 7;
      static const size_t attr_index_float  = 0;
      static const size_t attr_index_uint8  = 1;
      static const size_t attr_index_int8   = 2;
      static const size_t attr_index_uint16 = 3;
      static const size_t attr_index_int16  = 4;
      static const size_t attr_index_uint32 = 5;
      static const size_t attr_index_int32  = 6;

      std::vector < std::vector <float> >    float_values;
      std::vector < std::vector <uint8_t> >  uint8_values;
      std::vector < std::vector <int8_t> >   int8_values;
      std::vector < std::vector <uint16_t> > uint16_values;
      std::vector < std::vector <int16_t> >  int16_values;
      std::vector < std::vector <uint32_t> > uint32_values;
      std::vector < std::vector <int32_t> >  int32_values;
    
      // This method lets cereal know which data members to serialize
      template<class Archive>
      void serialize(Archive & archive)
      {
        archive( float_values,
                 uint8_values,
                 int8_values,
                 uint16_values,
                 int16_values,
                 uint32_values,
                 int32_values ); // serialize things by passing them to the archive
      }

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

      size_t insert (const std::vector<int8_t> &value)
      {
        size_t index;
        index = uint8_values.size();
        int8_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<uint16_t> &value)
      {
        size_t index;
        index = uint16_values.size();
        uint16_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<int16_t> &value)
      {
        size_t index;
        index = int16_values.size();
        int16_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<uint32_t> &value)
      {
        size_t index;
        index = uint32_values.size();
        uint32_values.push_back(value);
        return index;
      }

      size_t insert (const std::vector<int32_t> &value)
      {
        size_t index;
        index = int32_values.size();
        int32_values.push_back(value);
        return index;
      }

      template<class T>
      void push_back (size_t vindex, T value);

      template<class T>
      const T at (size_t vindex, size_t index) const;

      void append (AttrVal a)
      {
        assert(float_values.size()  == a.float_values.size());
        assert(uint8_values.size()  == a.uint8_values.size());
        assert(uint16_values.size() == a.uint16_values.size());
        assert(uint32_values.size() == a.uint32_values.size());
        assert(int8_values.size()   == a.int8_values.size());
        assert(int16_values.size()  == a.int16_values.size());
        assert(int32_values.size()  == a.int32_values.size());
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
        for (size_t i=0; i<int8_values.size(); i++)
          {
            int8_values[i].insert(int8_values[i].end(),
                                   a.int8_values[i].begin(),
                                   a.int8_values[i].end());
          }
        for (size_t i=0; i<uint16_values.size(); i++)
          {
            uint16_values[i].insert(uint16_values[i].end(),
                                   a.uint16_values[i].begin(),
                                   a.uint16_values[i].end());
          }
        for (size_t i=0; i<int16_values.size(); i++)
          {
            int16_values[i].insert(int16_values[i].end(),
                                   a.int16_values[i].begin(),
                                   a.int16_values[i].end());
          }
        for (size_t i=0; i<uint32_values.size(); i++)
          {
            uint32_values[i].insert(uint32_values[i].end(),
                                   a.uint32_values[i].begin(),
                                   a.uint32_values[i].end());
          }
        for (size_t i=0; i<int32_values.size(); i++)
          {
            int32_values[i].insert(int32_values[i].end(),
                                   a.int32_values[i].begin(),
                                   a.int32_values[i].end());
          }

      }
    };

    

    struct NamedAttrVal : AttrVal
    {

      std::map<std::string, size_t> float_names;
      std::map<std::string, size_t> uint8_names;
      std::map<std::string, size_t> int8_names;
      std::map<std::string, size_t> uint16_names;
      std::map<std::string, size_t> int16_names;
      std::map<std::string, size_t> uint32_names;
      std::map<std::string, size_t> int32_names;

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

      size_t insert (std::string name, const std::vector<int8_t> &value)
      {
        size_t index;
        index = int8_values.size();
        int8_values.push_back(value);
        int8_names.insert(make_pair(name, index));
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

      size_t insert (std::string name, const std::vector<int16_t> &value)
      {
        size_t index;
        index = int16_values.size();
        int16_values.push_back(value);
        int16_names.insert(make_pair(name, index));
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

      size_t insert (std::string name, const std::vector<int32_t> &value)
      {
        size_t index;
        index = int32_values.size();
        int32_values.push_back(value);
        int32_names.insert(make_pair(name, index));
        return index;
      }

      void attr_names (std::vector<std::vector<std::string> > &) const; 

    };
  }
}

#endif
