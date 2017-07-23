// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file bcast_string_vector.cc
///
///  Function for broadcasting a string vector via MPI.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================

#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>


namespace neuroh5
{

  namespace mpi
  {

    int bcast_string_vector (MPI_Comm comm, int root,
                             const size_t max_string_len,
                             std::vector<std::string> &string_vector);
  }
}
