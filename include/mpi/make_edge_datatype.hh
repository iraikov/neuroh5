// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file make_edge_datatype.hh
///
///  Function for creating MPI datatype for edge representation.
///
///  Copyright (C) 2017 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>


namespace neuroh5
{

  namespace mpi
  {

    MPI_Datatype make_edge_datatype (size_t num_float_attr,
                                     size_t num_uint8_attr,
                                     size_t num_uint16_attr,
                                     size_t num_uint32);

  }
}
