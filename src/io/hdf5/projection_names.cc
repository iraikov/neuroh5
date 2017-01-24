// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file projection_names.cc
///
///  Functions for reading projection names from an HDF5 file.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================
#include <mpi.h>
#include <hdf5.h>

#include <cstring>
#include <vector>

#include "debug.hh"
#include "hdf5_path_names.hh"
#include "bcast_string_vector.hh"


#undef NDEBUG
#include <cassert>

#define MAX_PRJ_NAME 1024
#define MAX_EDGE_ATTR_NAME 1024

using namespace std;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      //////////////////////////////////////////////////////////////////////////
      herr_t iterate_cb
      (
       hid_t             grp,
       const char*       name,
       const H5L_info_t* info,
       void*             op_data
       )
      {
        vector<string>* ptr = (vector<string>*)op_data;
        ptr->push_back(string(name));
        return 0;
      }

      //////////////////////////////////////////////////////////////////////////
      herr_t read_projection_names
      (
       MPI_Comm             comm,
       const std::string&   file_name,
       vector<string>&      prj_names
       )
      {
        herr_t ierr = 0;

        int rank, size;

        assert(MPI_Comm_size(comm, &size) >= 0);
        assert(MPI_Comm_rank(comm, &rank) >= 0);

        // MPI rank 0 reads and broadcasts the number of ranges
        hsize_t num_projections;
        hid_t file = -1, grp = -1;

        // MPI rank 0 reads and broadcasts the projection names
        if (rank == 0)
          {
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);

            grp = H5Gopen(file, PRJ.c_str(), H5P_DEFAULT);
            assert(grp >= 0);
            assert(H5Gget_num_objs(grp, &num_projections)>=0);

            hsize_t idx = 0;
            vector<string> op_data;
            assert(H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                              &iterate_cb, (void*)&op_data ) >= 0);

            assert(op_data.size() == num_projections);

            for (size_t i = 0; i < op_data.size(); ++i)
              {
                DEBUG("Projection ",i," is named ",op_data[i],"\n");
                prj_names[i] = op_data[i];
              }

            assert(H5Gclose(grp) >= 0);
            assert(H5Fclose(file) >= 0);
          }

        // Broadcast projection names
        ierr = mpi::bcast_string_vector(comm, MAX_PRJ_NAME, prj_names);

        return ierr;
      }
    }
  }
}
