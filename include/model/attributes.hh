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

#include "edge_attr.hh"
#include "hdf5_types.hh"
#include "ngh5_types.hh"

#include "hdf5.h"

#include <string>
#include <utility>
#include <vector>

namespace ngh5
{
  namespace model
  {
    /// @brief Discovers the list of edge attributes.
    ///
    /// @param file_name      Input file name
    ///
    /// @param proj_name      The (abbreviated) name of the projection data set.
    ///
    /// @param out_attributes    A vector of pairs, one for each edge attribute
    ///                          discovered. The pairs contain the attribute
    ///                          name and the attributes HDF5 file datatype.
    ///                          NOTE: The datatype handles MUST be closed by
    ///                          the caller (via H5Tclose).
    ///
    /// @return                  HDF5 error code.
    herr_t get_edge_attributes
    (
     const std::string&                           file_name,
     const std::string&                           proj_name,
     std::vector< std::pair<std::string,hid_t> >& out_attributes
     );

    /// @brief Determines the number of edge attributes for each supported type.
    ///
    ///
    /// @param attributes    A vector of pairs, one for each edge attribute
    ///                      discovered. The pairs contain the attribute name
    ///                      and the attributes HDF5 file datatype.
    ///
    /// @param num_attrs     A vector which indicates the number of attributes of each type.
    ///                      - Index 0 float type
    ///                      - Index 1: uint8/enum type
    ///                      - Index 1: uint16 type
    ///                      - Index 1: uint32 type
    ///
    /// @return                  HDF5 error code.
    herr_t num_edge_attributes
    (
     const std::vector< std::pair<std::string,hid_t> >& attributes,
     std:: vector <size_t> &num_attrs
     );

    /// @brief Reads the values of edge attributes.
    ///
    /// @param file_name      Input file name
    ///
    /// @param proj_name      The (abbreviated) name of the projection.
    ///
    /// @param attr_name      The name of the attribute.
    ///
    /// @param edge_base      Edge offset (returned by read_dbs_projection).
    ///
    /// @param edge_count     Edge count.
    ///
    /// @param attr_h5type    The HDF5 type of the attribute.
    ///
    /// @param attr_values    An EdgeNamedAttr object that holds attribute values.
    herr_t read_edge_attributes
    (
     MPI_Comm            comm,
     const std::string&  file_name,
     const std::string&  proj_name,
     const std::string&  attr_name,
     const DST_PTR_T     edge_base,
     const DST_PTR_T     edge_count,
     const hid_t         attr_h5type,
     EdgeNamedAttr      &attr_values
     );
  }
}

#endif
