// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file hdf5_path_names.cc
///
///  Definitions of Neurograph HDF5 path manipulation function.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "hdf5_path_names.hh"

using namespace std;

namespace neurotrees
{
  
  string cell_attribute_prefix
  (
   const string& name_space,
   const string& pop_name
   )
  {
    return "/" + POPS + "/" + pop_name + "/" + name_space;
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
  
  string population_path
  (
   const string& pop_name
   )
  {
    return "/" + POPS + "/" + pop_name;
  }
  
  string population_trees_path
  (
   const string& pop_name
   )
  {
    return "/" + POPS + "/" + pop_name + "/" + TREES;
  }
  
  string h5types_path_join(const string& name)
  {
    return H5_TYPES + "/" + name;
  }
  
}
