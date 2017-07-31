// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file group_contents.hh
///
///  Reads the contents of a group.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#ifndef GROUP_CONTENTS_HH
#define GROUP_CONTENTS_HH

#include <hdf5.h>
#include <mpi.h>

#include <string>
#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    /// @brief Reads the names of the elements of a group
    ///
    /// @param comm          MPI communicator
    ///
    /// @param file_name     Input file name
    ///
    /// @param path          Path to group
    ///
    /// @return              HDF5 error code
    extern herr_t group_contents
    (
     MPI_Comm                   comm,
     const hid_t&               file,
     const std::string&         path,
     std::vector <std::string>& obj_names
     );

    extern herr_t group_contents_serial
    (
     const hid_t&               file,
     const std::string&         path,
     std::vector <std::string>& obj_names
     );
}
}

#endif
