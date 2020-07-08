// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file group_contents.cc
///
///  Functions for reading the names of objects in a group.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================
#include <mpi.h>
#include <hdf5.h>

#include <cstring>
#include <vector>

#include "debug.hh"
#include "path_names.hh"

#include "throw_assert.hh"

using namespace std;

namespace neuroh5
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
      herr_t group_contents
      (
       MPI_Comm             comm,
       const hid_t&         file,
       const std::string&   path,
       vector<string>&      obj_names
       )
      {
        herr_t ierr = 0;

        int rank, size;

        throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                     "hdf5::group_contents: unable to obtain MPI communicator size");

        throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                     "hdf5::group_contents: unable to obtain MPI communicator rank");

        // MPI rank 0 reads and broadcasts the number of ranges
        hsize_t num_objs;
        hid_t grp = -1;

        // MPI rank 0 reads the object names
        if (rank == 0)
          {
            grp = H5Gopen(file, path.c_str(), H5P_DEFAULT);
            throw_assert(grp >= 0, "hdf5::group_contents: unable to open group " << path);
            throw_assert(H5Gget_num_objs(grp, &num_objs)>=0,
                         "hdf5::group_contents: unable to get number of objects in group " << path);
            hsize_t idx = 0;
            vector<string> op_data;
            throw_assert(H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                                    &iterate_cb, (void*)&op_data ) >= 0,
                         "hdf5::group_contents: unable to iterate over objects in group " << path);

            throw_assert(op_data.size() == num_objs, 
                         "hdf5::group_contents: group size mismatch");
            
            for (size_t i = 0; i < op_data.size(); ++i)
              {
                throw_assert(op_data[i].size() > 0,
                             "hdf5::group_contents: invalid object name in group " << path);
                
                obj_names.push_back(op_data[i]);
              }

            throw_assert(H5Gclose(grp) >= 0,
                         "hdf5::group_contents: unable to close group " << path);
          }

        return ierr;
      }
    }
}
