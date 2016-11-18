// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file edge_attr.hh
///
///  Functions for storing node and edge attributes in vectors of different
///  types.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef EDGE_ATTR_HH
#define EDGE_ATTR_HH

#include <map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <typeindex>

namespace ngh5
{
  namespace model
  {
    struct EdgeAttr
    {
      std::vector < std::vector <float> > float_values;
      std::vector < std::vector <uint8_t> > uint8_values;
      std::vector < std::vector <uint16_t> > uint16_values;
      std::vector < std::vector <uint32_t> > uint32_values;

      template<class T>
      const size_t size_attr_vec () const
      {
        size_t result = 0;
        if (typeid(T) == typeid(float))
          {
            result = float_values.size();
          }
        else
          if (std::type_index(typeid(T)) == std::type_index(typeid(uint8_t)))
            {
              result = uint8_values.size();
            }
          else if (std::type_index(typeid(T)) ==
                   std::type_index(typeid(uint16_t)))
            {
              result = uint16_values.size();
            }
          else if (std::type_index(typeid(T)) ==
                   std::type_index(typeid(uint32_t)))
            {
              result = uint32_values.size();
            }
          else
            throw std::runtime_error("Unknown type for size_attr_vec");
        return result;
      }

      template<class T>
      const void resize (size_t size)
      {
        if (typeid(T) == typeid(float))
          float_values.resize(size);
        else
          if (typeid(T) == typeid(uint8_t))
            uint8_values.resize(size);
          else if (typeid(T) == typeid(uint16_t))
            uint16_values.resize(size);
          else if (typeid(T) == typeid(uint32_t))
            uint32_values.resize(size);
          else
            throw std::runtime_error("Unknown type for resize");
      }


      template<class T>
      const void *attr_ptr (size_t i) const
      {
        const void *result;
        if (typeid(T) == typeid(float))
          result = &float_values[i][0];
        else
          if (typeid(T) == typeid(uint8_t))
            result = &uint8_values[i][0];
          else if (typeid(T) == typeid(uint16_t))
            result = &uint16_values[i][0];
          else if (typeid(T) == typeid(uint32_t))
            result = &uint32_values[i][0];
          else
            throw std::runtime_error("Unknown type for attr_ptr");
        return result;
      }

      template<class T>
      size_t size_attr (size_t i) const
      {
        size_t result;
        if (typeid(T) == typeid(float))
          result = float_values[i].size();
        else
          if (typeid(T) == typeid(uint8_t))
            result = uint8_values[i].size();
          else if (typeid(T) == typeid(uint16_t))
            result = uint16_values[i].size();
          else if (typeid(T) == typeid(uint32_t))
            result = uint32_values[i].size();
          else
            throw std::runtime_error("Unknown type for size_attr");
        return result;
      }

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
      void push_back (size_t vindex, T value)
      {
        if (typeid(T) == typeid(float))
          {
            float_values[vindex].push_back(value);
          }
        else
          if (typeid(T) == typeid(uint8_t))
            {
              uint8_values[vindex].push_back (value);
            }
          else if (typeid(T) == typeid(uint16_t))
            {
              uint16_values[vindex].push_back (value);
            }
          else if (typeid(T) == typeid(uint32_t))
            {
              uint32_values[vindex].push_back (value);
            }
          else
            throw std::runtime_error("Unknown type for push_back");
      }

      template<class T>
      const T at (size_t vindex, size_t index) const
      {
        T result;
        if (typeid(T) == typeid(float))
          {
            result = float_values[vindex][index];
          }
        else
          if (typeid(T) == typeid(uint8_t))
            {
              result = uint8_values[vindex][index];
            }
          else if (typeid(T) == typeid(uint16_t))
            {
              result = uint16_values[vindex][index];
            }
          else if (typeid(T) == typeid(uint32_t))
            {
              result = uint32_values[vindex][index];
            }
          else
            throw std::runtime_error("Unknown type for push_back");
        return result;
      }

      void append (EdgeAttr a)
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

    struct EdgeNamedAttr : EdgeAttr
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
}

#endif
