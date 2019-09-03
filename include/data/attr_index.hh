// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_index.hh
///
///  Functions for indexing attributes of different types.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================

#ifndef ATTR_INDEX_HH
#define ATTR_INDEX_HH

#include <map>
#include <vector>
#include <set>
#include <string>
#include <typeindex>
#include <iterator>

#include "throw_assert.hh"

#include "cereal/types/vector.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"
#include "cereal/types/memory.hpp"

namespace neuroh5
{
  namespace data
  {

    
    struct AttrSet
    {
  
      std::map<std::type_index, std::set <std::string> > attr_name_map;
  
      AttrSet()
      {
      };

      // This method lets cereal know which data members to serialize
      template<class Archive>
      void serialize(Archive & archive)
      {
        // serialize things by passing them to the archive
        archive(attr_name_map);
          
      }

      template<class T>
      size_t attr_type_index ()
      {
        auto first = this->attr_name_map.begin();
        auto entry = this->attr_name_map[std::type_index(typeid(T))];
        return std::distance(first, this->attr_name_map.find(std::type_index(typeid(T))));
      }

      template<class T>
      size_t size_attr_names () 
      {
        return this->attr_name_map[std::type_index(typeid(T))].size();
      }

      
      template<class T>
      const std::vector<std::string> attr_names () 
      {
        std::vector<std::string> output;
        auto entry = this->attr_name_map[std::type_index(typeid(T))];
        auto it = this->attr_name_map.find(std::type_index(typeid(T)));
        std::copy(it->second.begin(), it->second.end(), std::back_inserter(output));
        return output;
      }

      template<class T>
      void add (const std::string& name)
      {
        this->attr_name_map[std::type_index(typeid(T))].insert(name);
      }

    };

    struct AttrIndex
    {
      std::map<std::type_index, std::map <std::string, size_t> > attr_name_map;

      AttrIndex() {};
      
      AttrIndex(const AttrSet& attr_names_set)
      {

        for (auto attr_type_it=attr_names_set.attr_name_map.begin();
             attr_type_it!=attr_names_set.attr_name_map.end();
             ++attr_type_it)
          {
            const std::set<std::string>& attr_names = attr_type_it->second;
            auto first = attr_names.begin();
            std::map <std::string, size_t>& this_map = attr_name_map[attr_type_it->first];
            for (auto attr_name_it=attr_names.cbegin();
                 attr_name_it != attr_names.cend();
                 ++attr_name_it)
              {
                size_t idx = std::distance(first, attr_name_it);
                this_map.insert(make_pair(*attr_name_it, idx));
              }
              
          }
        
      };


      // This method lets cereal know which data members to serialize
      template<class Archive>
      void serialize(Archive & archive)
      {
        // serialize things by passing them to the archive
        archive(attr_name_map);
          
      }

      template<class T>
      size_t attr_type_index ()
      {
        auto first = this->attr_name_map.begin();
        auto entry = this->attr_name_map[std::type_index(typeid(T))];
        return std::distance(first, this->attr_name_map.find(std::type_index(typeid(T))));
      }

      template<class T>
      size_t size_attr_index () const
      {
        auto type_it = this->attr_name_map.find(std::type_index(typeid(T)));
        if (type_it != this->attr_name_map.cend())
          {
            return type_it->second.size();
          }
        else
          {
            return 0;
          }
      }
      
      template<class T>
      const std::vector<std::string> attr_names () const
      {
        std::vector<std::string> output;
        auto type_it = this->attr_name_map.find(std::type_index(typeid(T)));
        if (type_it != this->attr_name_map.cend())
          {
            const std::map <std::string, size_t>& attr_items = type_it->second;
            for (auto item_it=attr_items.cbegin(); item_it!=attr_items.cend(); ++item_it)
              {
                output.push_back(item_it->first);
              }
          }
        return output;
      }
      
      template<class T>
      size_t attr_index (const std::string attr_name) const
      {
        auto type_it  = this->attr_name_map.find(std::type_index(typeid(T)));
        if (type_it != this->attr_name_map.cend())
          {
            const std::map <std::string, size_t>& m = type_it->second;
            auto it = m.find(attr_name);
            if (it != m.end())
              {
                return it->second;
              }
            else
              {
                throw std::runtime_error("AttrIndex::attr_index: unknown attribute");
              }
          }
        else
          {
            throw std::runtime_error("AttrIndex::attr_index: unknown type");
          }
      }
      

      template<class T>
      void insert (const std::string& name, size_t index)
      {
        this->attr_name_map[std::type_index(typeid(T))].insert(make_pair(name, index));
      }

    };



  }
}

#endif
