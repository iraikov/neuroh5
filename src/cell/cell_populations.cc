// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_populations.cc
///
///  Functions for reading population names from an HDF5 enumerated type.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
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

namespace neuroio
{

  namespace cell
  {
  
    //////////////////////////////////////////////////////////////////////////
    herr_t enum_population_names
    (
     MPI_Comm             comm,
     hid_t                file,
     vector<string>&      pop_enum_names
     )
    {
      herr_t ierr = 0;
    
      int rank, size;
    
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);
    
      // MPI rank 0 reads and broadcasts the names of populations
      hid_t ty = -1;

      if (rank == 0)
        {
          ty = H5Topen( file, h5types_path_join(POPLABELS).c_str(), H5P_DEFAULT);
          assert(ty >= 0);

          int num_members = H5Tget_nmembers(ty);
        
          for (size_t i = 0; i < num_members; ++i)
            {
              char cname[MAX_POP_NAME_LEN]; string name;
              ierr = H5Tenum_nameof(ty, &i, cname, MAX_POP_NAME_LEN);
              assert(ierr >= 0);
              name = string(cname);
              pop_enum_names.push_back(name);
            }
        
          assert(H5Tclose(ty) >= 0);
        }

      ierr = mpi_bcast_string_vector (comm,
                                      MAX_POP_NAME_LEN,
                                      pop_enum_names);
      return ierr;
    }

  }
}
