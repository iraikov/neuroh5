// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file projection_names.cc
///
///  Reads the projection names
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#ifndef PROJECTION_NAMES_HH
#define PROJECTION_NAMES_HH

#include "mpi.h"

#include <string>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      /// @brief Reads the names of projections
      ///
      /// @param comm          MPI communicator
      ///
      /// @param file_name     Input file name
      ///
      /// @param proj_names    Vector of projection names
      ///
      /// @return              HDF5 error code
      extern herr_t read_projection_names
      (
       MPI_Comm                  comm,
       const std::string&        file_name,
       std::vector<std::string>& proj_names
       );
    }
  }
}

#endif
