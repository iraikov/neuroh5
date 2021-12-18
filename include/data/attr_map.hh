// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.hh
///
///  Functions for storing attributes in vectors of different types.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#ifndef ATTR_MAP_HH
#define ATTR_MAP_HH

#include <map>
#include <set>
#include <vector>
#include <deque>
#include <typeindex>

#include "throw_assert.hh"
#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace data
  {

    template<typename T>
    void append_values_map(std::map < CELL_IDX_T, std::deque <T> >& values_map,
                           CELL_IDX_T vindex,
                           typename vector<T>::const_iterator first,
                           typename vector<T>::const_iterator last)
    {
      if (last > first)
        {
          auto values_it = values_map.find(vindex);
          if (values_it == values_map.end())
            {
              deque<T> v(first, last);
              values_map.insert(make_pair(vindex, v));
            }
          else
            {
              deque<T>& v = values_it->second;
              v.insert(std::end(v), first, last);
            }
        }
    }

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

      std::vector <std::map < CELL_IDX_T, std::deque <float> > >    float_values;
      std::vector <std::map < CELL_IDX_T, std::deque <uint8_t> > >  uint8_values;
      std::vector <std::map < CELL_IDX_T, std::deque <int8_t> > >   int8_values;
      std::vector <std::map < CELL_IDX_T, std::deque <uint16_t> > > uint16_values;
      std::vector <std::map < CELL_IDX_T, std::deque <int16_t> > >  int16_values;
      std::vector <std::map < CELL_IDX_T, std::deque <uint32_t> > > uint32_values;
      std::vector <std::map < CELL_IDX_T, std::deque <int32_t> > >  int32_values;

      typedef std::map < CELL_IDX_T, std::deque <float> > float_values_map;
      typedef std::map < CELL_IDX_T, std::deque <uint8_t> > uint8_values_map;
      typedef std::map < CELL_IDX_T, std::deque <uint16_t> > uint16_values_map;
      typedef std::map < CELL_IDX_T, std::deque <uint32_t> > uint32_values_map;
      typedef std::map < CELL_IDX_T, std::deque <int8_t> > int8_values_map;
      typedef std::map < CELL_IDX_T, std::deque <int16_t> > int16_values_map;
      typedef std::map < CELL_IDX_T, std::deque <int32_t> > int32_values_map;

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
      const std::map<CELL_IDX_T, std::deque<T> >& attr_map (size_t i) const; 
      template<class T>
      std::map<CELL_IDX_T, std::deque<T> >& attr_map (size_t i);
      
      template<class T>
      const vector< std::map<CELL_IDX_T, std::deque<T> > >& attr_maps () const; 
      template<class T>
      vector< std::map<CELL_IDX_T, std::deque<T> > >& attr_maps (); 

      void num_attrs (vector<size_t> &v) const;

      template<class T>
      size_t num_attr () const
      {
        const std::vector <std::map<CELL_IDX_T, std::deque<T> > >& value_map = attr_maps<T>();
        return value_map.size();
      }

      template<class T>
      size_t insert (const std::vector<CELL_IDX_T> &cell_index,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<T> &value)
      {
        std::vector <std::map<CELL_IDX_T, std::deque<T> > >& value_map = attr_maps<T>();
        size_t attr_index = value_map.size();
        value_map.resize(max(value_map.size(), attr_index+1));
        for (size_t p=0; p<cell_index.size(); p++)
          {
            CELL_IDX_T vindex = cell_index[p];
            typename std::vector<T>::const_iterator first, last;
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
            append_values_map(value_map[attr_index], vindex, first, last);
            index_set.insert(vindex);
          }
        return attr_index;
      }


      template<class T>
      size_t insert (const size_t attr_index,
                     const std::vector<CELL_IDX_T> &cell_index,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<T> &value)
      {
        std::vector <std::map<CELL_IDX_T, std::deque<T> > >& value_map = attr_maps<T>();
        value_map.resize(max(value_map.size(), attr_index+1));
        for (size_t p=0; p<cell_index.size(); p++)
          {
            CELL_IDX_T vindex = cell_index[p];
            typename vector<T>::const_iterator first, last;
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
            append_values_map(value_map[attr_index], vindex, first, last);
            index_set.insert(vindex);
          }
        return attr_index;
      }

      
      template<class T>
      size_t insert (const size_t attr_index,
                     const CELL_IDX_T &cell_index,
                     const std::deque<T> &value)
      {
        std::vector <std::map<CELL_IDX_T, std::deque<T> > >& attr_values = attr_maps<T>();
        attr_values.resize(max(attr_values.size(), attr_index+1));
        attr_values[attr_index].insert(make_pair(cell_index, value));
        index_set.insert(cell_index);
        return attr_index;
      }

      

      template<class T>
      const vector<deque<T>> find (CELL_IDX_T index) const
      {
        const std::vector <std::map<CELL_IDX_T, std::deque<T> > >& attr_values = attr_maps<T>();
        vector< deque<T> > result;
        for (size_t i=0; i<attr_values.size(); i++)
          {
            auto it = attr_values[i].find(index);
            if (it != attr_values[i].end())
              {
                result.push_back(it->second);
              }
          }
        return result;
      }

      template<class T>
      void insert_map1 (vector <map <CELL_IDX_T, std::deque<T> > >& a,
                        const vector <map <CELL_IDX_T, std::deque<T> > >& b)
      {
        throw_assert(a.size() == b.size(), 
                     "AttrMap::insert_map1: maps are of different sizes: size a = " <<
                     a.size() << " size b = " << b.size());
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

      std::map<std::type_index, std::map <std::string, size_t> > attr_name_map;

      template<class T>
      void attr_names_type (std::vector<std::string> &output) const
      {
        auto type_it = attr_name_map.find(std::type_index(typeid(T)));
        if (type_it != attr_name_map.cend())
          {
            
            const map< string, size_t> &attr_names = type_it->second;
            output.resize(attr_names.size());
            for (auto element : attr_names)
              {
                output[element.second] = element.first;
              }
          }
      }
      
      void attr_names (std::vector<std::vector<std::string> > &) const; 

      template<class T>
      size_t insert_name (std::string name)
      {
        std::map <std::string, size_t>& name_map = this->attr_name_map[std::type_index(typeid(T))];
        std::vector <std::map<CELL_IDX_T, std::deque<T> > >& value_map = attr_maps<T>();
        size_t index = 0;
        if (name_map.count(name) == 0)
          {
            index = name_map.size();
            name_map.insert(make_pair(name, index));
            value_map.resize(max(value_map.size(), index+1));
          }
        else
          {
            auto it = name_map.find(name);
            index = it->second;
          }

        return index;
      }

      template<class T>
      size_t insert (std::string name, 
                     const std::vector<CELL_IDX_T> &cell_index,
                     const std::vector<ATTR_PTR_T> &ptr,
                     const std::vector<T> &value)
      {
        size_t attr_index = insert_name<T>(name);
        AttrMap::insert(attr_index, cell_index, ptr, value);
        return attr_index;
      }

      template<class T>
      const deque<T> find_name (const std::string& name, CELL_IDX_T& index)
      {
        deque<T> result(0);
        auto type_it = attr_name_map.find(std::type_index(typeid(T)));
        if (type_it != this->attr_name_map.cend())
          {
            const std::map <std::string, size_t>& name_map = type_it->second;

            auto attr_it = name_map.find(name);
            throw_assert(attr_it != name_map.end(),
                         "NamedAttrMap::find_name: attribute " << name << " not found");
            size_t attr_index = attr_it->second;
            std::map<CELL_IDX_T, std::deque<T> >& value_map = attr_map<T>(attr_index);
            auto it = value_map.find(index);
            if (it != value_map.end())
              {
                std::copy(it->second.begin(), it->second.end(), back_inserter(result)); 
              }
          }
        return result;
      }

      void clear()
      {
        attr_name_map.clear();
        AttrMap::clear();
      }

    };
  }
}

#endif
