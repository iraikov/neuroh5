// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.hh
///
///  Functions for storing attributes in vectors of different types.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef ATTR_MAP_HH
#define ATTR_MAP_HH

#include <map>
#include <set>
#include <vector>

// type support
#include "cereal/types/vector.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/set.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"
#include <cereal/types/memory.hpp>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace data
  {

    struct AttrMap
    {
      
      static const size_t num_attr_types    = 7;
      static const size_t attr_index_float  = 0;
      static const size_t attr_index_uint8  = 1;
      static const size_t attr_index_int8   = 2;
      static const size_t attr_index_uint16 = 3;
      static const size_t attr_index_int16  = 4;
      static const size_t attr_index_uint32 = 5;
      static const size_t attr_index_int32  = 6;
      
      std::set<CELL_IDX_T> index_set;
      std::vector <std::map < CELL_IDX_T, std::vector <float> > >    float_values;
      std::vector <std::map < CELL_IDX_T, std::vector <uint8_t> > >  uint8_values;
      std::vector <std::map < CELL_IDX_T, std::vector <int8_t> > >   int8_values;
      std::vector <std::map < CELL_IDX_T, std::vector <uint16_t> > > uint16_values;
      std::vector <std::map < CELL_IDX_T, std::vector <int16_t> > >  int16_values;
      std::vector <std::map < CELL_IDX_T, std::vector <uint32_t> > > uint32_values;
      std::vector <std::map < CELL_IDX_T, std::vector <int32_t> > >  int32_values;

      typedef std::map < CELL_IDX_T, std::vector <float> > float_values_map;
      typedef std::map < CELL_IDX_T, std::vector <uint8_t> > uint8_values_map;
      typedef std::map < CELL_IDX_T, std::vector <uint16_t> > uint16_values_map;
      typedef std::map < CELL_IDX_T, std::vector <uint32_t> > uint32_values_map;
      typedef std::map < CELL_IDX_T, std::vector <int8_t> > int8_values_map;
      typedef std::map < CELL_IDX_T, std::vector <int16_t> > int16_values_map;
      typedef std::map < CELL_IDX_T, std::vector <int32_t> > int32_values_map;

      typedef float_values_map::iterator float_values_iter;
      typedef uint8_values_map::iterator uint8_values_iter;
      typedef uint16_values_map::iterator uint16_values_iter;
      typedef uint32_values_map::iterator uint32_values_iter;
      typedef int8_values_map::iterator int8_values_iter;
      typedef int16_values_map::iterator int16_values_iter;
      typedef int32_values_map::iterator int32_values_iter;

      typedef float_values_map::const_iterator float_values_const_iter;
      typedef uint8_values_map::const_iterator uint8_values_const_iter;
      typedef uint16_values_map::const_iterator uint16_values_const_iter;
      typedef uint32_values_map::const_iterator uint32_values_const_iter;
      typedef int8_values_map::const_iterator int8_values_const_iter;
      typedef int16_values_map::const_iterator int16_values_const_iter;
      typedef int32_values_map::const_iterator int32_values_const_iter;

      // This method lets cereal know which data members to serialize
      template<class Archive>
      void serialize(Archive & archive)
      {
        archive(index_set,
                float_values,
                uint8_values,
                int8_values,
                uint16_values,
                int16_values,
                uint32_values,
                int32_values); // serialize things by passing them to the archive
      }

      template<class T>
      size_t insert (const std::vector<CELL_IDX_T> &index,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<T> &value);
      template<class T>
      size_t insert (const size_t index,
                     const CELL_IDX_T &cell_index,
                     const std::vector<T> &value);

      
      template<class T>
      const std::map<CELL_IDX_T, std::vector<T> >& attr_map (size_t i) const; 
      template<class T>
      const vector< std::map<CELL_IDX_T, std::vector<T> > >& attr_maps () const; 

      void num_attrs (vector<size_t> &v) const;

      template<class T>
      size_t num_attr () const;

      template<class T>
      const vector<vector<T>> find (CELL_IDX_T index) const;

      template<class T>
      void insert_map1 (vector <map <CELL_IDX_T, std::vector<T> > >& a,
                        const vector <map <CELL_IDX_T, std::vector<T> > >& b)
      {
        assert(a.size() == b.size());
        for (size_t i=0; i<a.size(); i++)
          {
            a[i].insert(b[i].cbegin(), b[i].cend());
          }
      }
      
      void insert_map (AttrMap a)
      {
        index_set.insert(a.index_set.begin(), a.index_set.end());
        insert_map1(float_values, a.float_values);
        insert_map1(uint8_values, a.uint8_values);
        insert_map1(uint16_values, a.uint16_values);
        insert_map1(uint32_values, a.uint32_values);
        insert_map1(int8_values,  a.int8_values);
        insert_map1(int16_values, a.int16_values);
        insert_map1(int32_values, a.int32_values);
      }

      void append (AttrMap a)
      {
        index_set.insert(a.index_set.begin(),
                         a.index_set.end());
        float_values.insert(float_values.end(),
                            a.float_values.begin(),
                            a.float_values.end());
        uint8_values.insert(uint8_values.end(),
                            a.uint8_values.begin(),
                            a.uint8_values.end());
        int8_values.insert(int8_values.end(),
                           a.int8_values.begin(),
                           a.int8_values.end());
        uint16_values.insert(uint16_values.end(),
                             a.uint16_values.begin(),
                             a.uint16_values.end());
        int16_values.insert(int16_values.end(),
                            a.int16_values.begin(),
                            a.int16_values.end());
        uint32_values.insert(uint32_values.end(),
                             a.uint32_values.begin(),
                             a.uint32_values.end());
        int32_values.insert(int32_values.end(),
                            a.int32_values.begin(),
                            a.int32_values.end());
      }

      void erase (const CELL_IDX_T &index)
      {
        for (size_t i=0; i<float_values.size(); i++)
          {
            float_values[i].erase(index);
          }
        for (size_t i=0; i<uint8_values.size(); i++)
          {
            uint8_values[i].erase(index);
          }
        for (size_t i=0; i<int8_values.size(); i++)
          {
            int8_values[i].erase(index);
          }
        for (size_t i=0; i<uint16_values.size(); i++)
          {
            uint16_values[i].erase(index);
          }
        for (size_t i=0; i<int16_values.size(); i++)
          {
            int16_values[i].erase(index);
          }
        for (size_t i=0; i<uint32_values.size(); i++)
          {
            uint32_values[i].erase(index);
          }
        for (size_t i=0; i<int32_values.size(); i++)
          {
            int32_values[i].erase(index);
          }
        index_set.erase(index);
      }

      void clear ()
      {
        float_values.clear();
        uint8_values.clear();
        int8_values.clear();
        uint16_values.clear();
        int16_values.clear();
        uint32_values.clear();
        int32_values.clear();
        index_set.clear();
      }
      
    };

    

    struct NamedAttrMap : AttrMap
    {

      std::map<std::string, size_t> float_names;
      std::map<std::string, size_t> uint8_names;
      std::map<std::string, size_t> int8_names;
      std::map<std::string, size_t> uint16_names;
      std::map<std::string, size_t> int16_names;
      std::map<std::string, size_t> uint32_names;
      std::map<std::string, size_t> int32_names;

      template<class T>
      const std::map< std::string, size_t> & attr_name_map() const; 

      void attr_names (std::vector<std::vector<std::string> > &) const; 

      template<class T>
      void insert_name (std::string, size_t); 

      template<class T>
      size_t insert (std::string name,
                     const std::vector<CELL_IDX_T> &index,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<T> &value);
      template<class T>
      size_t insert (const size_t index,
                     const CELL_IDX_T &cell_index,
                     const std::vector<T> &value);

      template<class T>
      const vector<T> find_name (const std::string& name, CELL_IDX_T& index);

    };
  }
}

#endif
