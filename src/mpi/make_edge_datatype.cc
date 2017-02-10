// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file make_edge_datatype.cc
///
///  Function for creating MPI datatype for edge representation.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================
#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>
#include "ngh5_types.hh"

#undef NDEBUG
#include <cassert>

using namespace std;

namespace ngh5
{

  namespace mpi
  {

    MPI_Datatype make_edge_datatype (size_t num_float_attr,
                                     size_t num_uint8_attr,
                                     size_t num_uint16_attr,
                                     size_t num_uint32)
    {
      MPI_Datatype mpi_edge_type, mpi_edge_struct;
      int nitems = 2 + num_float_attr + num_uint8_attr + num_uint16_attr + num_uint32;
      vector <int> blocklengths(nitems,1);
      vector<MPI_Datatype> types;
      vector<MPI_Aint> offsets;
      MPI_Aint lb, extent, offset;
      
      types.push_back(NODE_IDX_MPI_T);
      types.push_back(NODE_IDX_MPI_T);
      for (size_t i=0; i<num_float_attr; i++)
        {
          types.push_back(MPI_FLOAT);
        }
      for (size_t i=0; i<num_uint8_attr; i++)
        {
          types.push_back(MPI_UINT8_T);
        }
      for (size_t i=0; i<num_uint16_attr; i++)
        {
          types.push_back(MPI_UINT16_T);
        }
      for (size_t i=0; i<num_uint32_attr; i++)
        {
          types.push_back(MPI_UINT32_T);
        }

      MPI_Type_get_extent(NODE_IDX_MPI_T, &lb, &extent);
      offset = 0;
      offsets.push_back(offset);
      offset += extent;
      offsets.push_back(offset);
      offset += extent;

      MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);
      for (size_t i=0; i<num_float_attr; i++)
        {
          offsets.push_back(offset);
          offset += extent;
        }
      MPI_Type_get_extent(MPI_UINT8_T, &lb, &extent);
      for (size_t i=0; i<num_uint8_attr; i++)
        {
          offsets.push_back(offset);
          offset += extent;
        }
      MPI_Type_get_extent(MPI_UINT16_T, &lb, &extent);
      for (size_t i=0; i<num_uint16_attr; i++)
        {
          offsets.push_back(offset);
          offset += extent;
        }
      MPI_Type_get_extent(MPI_UINT32_T, &lb, &extent);
      for (size_t i=0; i<num_uint32_attr; i++)
        {
          offsets.push_back(offset);
          offset += extent;
        }

      MPI_Type_create_struct(nitems, &blocklengths[0], &offsets[0], &types[0], &mpi_edge_struct);
      MPI_Type_commit(&mpi_edge_struct);

      MPI_Type_create_resized(mpi_edge_struct, 0, sizeof(edge), &mpi_edge_type);
      MPI_Type_commit(&mpi_edge_type);

      return mpi_edge_type;
    }

  }
}
