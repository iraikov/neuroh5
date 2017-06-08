// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file projection_names.cc
///
///  Functions for reading projection names from an HDF5 file.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================
#include <mpi.h>
#include <hdf5.h>

#include <cstring>
#include <vector>

#include "debug.hh"
#include "path_names.hh"
#include "group_contents.hh"
#include "bcast_string_vector.hh"


#undef NDEBUG
#include <cassert>

#define MAX_PRJ_NAME 1024

using namespace std;

namespace neuroh5
{
    namespace graph
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
       vector<pair<string,string>>&      prj_names
       )
      {
        herr_t ierr = 0;

        int rank, size;

        assert(MPI_Comm_size(comm, &size) >= 0);
        assert(MPI_Comm_rank(comm, &rank) >= 0);

        vector<string> prj_src_pop_names, prj_dst_pop_names;
        
        // MPI rank 0 reads and broadcasts the number of ranges
        hid_t file = -1, grp = -1;

        // MPI rank 0 reads and broadcasts the projection names
        if (rank == 0)
          {
            vector <string> dst_pop_names;
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);

            assert(hdf5::group_contents(file, hdf5::PROJECTIONS, dst_pop_names) >= 0);

            for (size_t i=0; i<dst_pop_names.size(); i++)
              {
                vector <string> src_pop_names;
                const string& dst_pop_name = dst_population_names[i];

                assert(hdf5::group_contents(file, hdf5::PROJECTIONS+"/"+dst_pop_name, src_pop_names) >= 0);

                for (size_t j=0; j<src_pop_names.size(); j++)
                  {
                    prj_src_pop_names.push_back(src_pop_names[j]);
                    prj_dst_pop_names.push_back(dst_pop_name);
                  }
              }

            assert(H5Fclose(file) >= 0);
          }

        // Broadcast projection names
        ierr = mpi::bcast_string_vector(comm, 0, MAX_PRJ_NAME, prj_src_pop_names);
        ierr = mpi::bcast_string_vector(comm, 0, MAX_PRJ_NAME, prj_dst_pop_names);

        for (size_t i=0; i<prj_dst_pop_names.size(); i++)
          {
            prj_names.push_back(make_pair(prj_src_pop_names[i],
                                          prj_dst_pop_names[i]));
          }

        return ierr;
      }
    }
}
