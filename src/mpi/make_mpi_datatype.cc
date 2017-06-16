// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file mpi_bcast_string_vector.cc
///
///  Function for broadcasting a string vector via MPI.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================
#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace ngh5
{

  namespace mpi
  {

    MPI_Datatype make_edge_datatype (MPI_Comm comm, 
                                     size_t num_float_attr,
                                     size_t num_uint8_attr,
                                     size_t num_uint16_attr,
                                     size_t num_uint32)
    {
      
    }

  }
}
