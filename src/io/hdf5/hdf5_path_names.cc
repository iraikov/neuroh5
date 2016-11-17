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

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      //////////////////////////////////////////////////////////////////////////
      string edge_attribute_path
      (
       const string& proj_name
       )
      {
        return PRJ + "/" + proj_name + "/" + ATTR + "/" + EDGE;
      }

      //////////////////////////////////////////////////////////////////////////
      string edge_attribute_path
      (
       const string& proj_name,
       const string& attr_name
       )
      {
        return edge_attribute_path(proj_name) + "/" + attr_name;
      }

      //////////////////////////////////////////////////////////////////////////
      string projection_path_join(const string& proj_name, const string& name)
      {
        return PRJ + "/" + proj_name + "/" + name;
      }

      //////////////////////////////////////////////////////////////////////////
      string h5types_path_join(const string& name)
      {
        return H5_TYPES + "/" + name;
      }
    }
  }
}
