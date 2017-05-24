// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_population_names.cc
///
///  Functions for reading population names from an HDF5 file.
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <hdf5.h>

#include <cstring>
#include <vector>

#include "mpi_bcast_string_vector.hh"
#include "hdf5_path_names.hh"

#undef NDEBUG
#include <cassert>

#define MAX_POP_NAME_LEN 1024

using namespace std;

namespace neurotrees
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
  herr_t read_population_names
  (
   MPI_Comm             comm,
   hid_t                file,
   vector<string>&      pop_names
   )
  {
    herr_t ierr = 0;
    
    int rank, size;
    
    assert(MPI_Comm_size(comm, &size) >= 0);
    assert(MPI_Comm_rank(comm, &rank) >= 0);
    
    // MPI rank 0 reads and broadcasts the names of populations
    hid_t grp = -1;

    // Rank 0 reads the names of populations and broadcasts
    if (rank == 0)
      {
        hsize_t num_populations;
        grp = H5Gopen(file, POPS.c_str(), H5P_DEFAULT);
        assert(grp >= 0);
        assert(H5Gget_num_objs(grp, &num_populations)>=0);

        hsize_t idx = 0;
        vector<string> op_data;
        assert(H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                          &iterate_cb, (void*)&op_data ) >= 0);
        
        assert(op_data.size() == num_populations);
        
        for (size_t i = 0; i < op_data.size(); ++i)
          {
            pop_names.push_back(op_data[i]);
          }
        
        assert(H5Gclose(grp) >= 0);
      }

    ierr = mpi_bcast_string_vector (comm,
                                    MAX_POP_NAME_LEN,
                                    pop_names);
    return ierr;
  }

}
