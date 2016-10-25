// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attributes.hh
///
///  Auxilliary functions for node and edge attribute discovery.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef ATTRIBUTES_HH
#define ATTRIBUTES_HH

#include "hdf5.h"
#include "ngh5types.hh"

#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace ngh5
{
  
  /// @brief Specifies the path to edge attributes
  ///
  /// @param dsetname          Projection data set name
  ///
  /// @param attr_name         Edge attribute name
  ///
  /// @return                  A string containing the full path to the attribute data set 
  std::string ngh5_edge_attr_path (const char *dsetname, const char *attr_name);

  /// @brief Discovers the list of edge attributes.
  ///
  /// @param in_file           Input file name
  ///
  /// @param in_projName       The (abbreviated) name of the projection.
  ///
  /// @param out_attributes    A vector of pairs, one for each edge attribute
  ///                          discovered. The pairs contain the attribute name
  ///                          and the attributes HDF5 file datatype.
  ///                          NOTE: The datatype handles MUST be closed by the
  ///                          caller (via H5Tclose).
  ///
  /// @return                  HDF5 error code.
  herr_t get_edge_attributes
  (
   const char*                                  in_file, 
   const std::string&                           in_projName,
   std::vector< std::pair<std::string,hid_t> >& out_attributes
   );

  
  /// @brief Reads the values of edge attributes.
  ///
  /// @param in_file           Input file name
  ///
  /// @param dsetname          The (abbreviated) name of the projection.
  herr_t read_edge_attributes
  (
   MPI_Comm            comm,
   const char*         in_file, 
   const char*         dsetname, 
   const char*         attr_name, 
   const DST_PTR_T     edge_base,
   const DST_PTR_T     edge_count,
   const hid_t         attr_h5type,
   EdgeAttr           &attr_values
   );

}

#endif
