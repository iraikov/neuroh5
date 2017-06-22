// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attr_map.hh
///
///  Functions for storing optional values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#ifndef OPTIONAL_VALUE_HH
#define OPTIONAL_VALUE_HH

#include <hdf5.h>

#include <map>
#include <set>
#include <vector>

namespace neuroh5
{
  namespace data
  {

    struct string_empty_value : compact_optional_type<std::string>
    {
      static std::string empty_value()
      {
        return std::string("\0\0", 2);
      }
      
      static bool is_empty_value(const std::string& v)
      {
        return v.compare(0, v.npos, "\0\0", 2) == 0;
      }
    };

    typedef compact_optional<string_empty_value> optional_string;
    
    struct hid_empty_value : compact_optional_type<hid_t>
    {
      static hid_t empty_value()
      {
        return -1;
      }
      
      static bool is_empty_value(const hid_t& v)
      {
        return v == -1;
      }
    };

    typedef compact_optional<hid_empty_value> optional_hid;


  }
}

#endif
