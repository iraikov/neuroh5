// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attributes.hh
///
///  Auxilliary functions for node and edge attribute discovery.
///
///  Copyright (C) 2016 Project Neurograh.
//==============================================================================

#ifndef ATTRIBUTES_HH
#define ATTRIBUTES_HH

#include "hdf5.h"

#include <string>
#include <utility>
#include <vector>

namespace ngh5
{

  /// @brief Discovers the list of edge attributes.
  ///
  /// @param in_file           HDF5 input file handle
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
   hid_t&                                       in_file,
   const std::string&                           in_projName,
   std::vector< std::pair<std::string,hid_t> >& out_attributes
   );

}

#endif
