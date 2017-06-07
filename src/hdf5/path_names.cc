// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file path_names.cc
///
///  Definitions of Neurograph HDF5 path manipulation function.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "path_names.hh"

using namespace std;

namespace neuroio
{
  namespace hdf5
  {
    
    string cell_attribute_prefix
    (
     const string& name_space,
     const string& pop_name
     )
    {
      return "/" + POPULATIONS + "/" + pop_name + "/" + name_space;
    }
    
    string cell_attribute_path
    (
     const string& name_space,
     const string& pop_name,
     const string& attr_name
     )
    {
      return cell_attribute_prefix(name_space, pop_name) + "/" + attr_name;
    }
    
    string projection_attribute_path
    (
     const string& src_pop_space,
     const string& dst_pop_name,
     const string& attr_name
     )
    {
      return "/" + PROJECTIONS + "/" + dst_pop_name + "/" + src_pop_name + "/" + attr_name;
    }
    
    string projection_attribute_path
    (
     const string& src_pop_name,
     const string& dst_pop_name,
     const string& attr_name
     )
    {
      return projection_attribute_prefix(src_pop_name, dst_pop_name) + "/" + attr_name;
    }
    
    string population_path
    (
     const string& pop_name
     )
    {
      return "/" + POPULATIONS + "/" + pop_name;
    }
    
    string h5types_path_join(const string& name)
    {
      return H5_TYPES + "/" + name;
    }
    
  }
}
