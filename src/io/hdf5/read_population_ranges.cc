// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_population_ranges.cc
///
///  Functions for reading population ranges from an HDF5 file.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <hdf5.h>

#include <cstring>
#include <vector>
#include <map>

#include "neurotrees_types.hh"
#include "hdf5_path_names.hh"

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neurotrees
{

  /*************************************************************************
   * Read the population ranges
   *************************************************************************/
  
  herr_t read_population_ranges
  (
   MPI_Comm                                comm,
   const std::string&                      file_name,
   map<CELL_IDX_T, pair<uint32_t,pop_t> >& pop_ranges,
   vector<pop_range_t> &pop_vector,
   size_t &n_nodes
   )
  {
    herr_t ierr = 0;
    
    int rank, size;
    assert(MPI_Comm_size(comm, &size) >= 0);
    assert(MPI_Comm_rank(comm, &rank) >= 0);

    // MPI rank 0 reads and broadcasts the number of ranges
    
    uint64_t num_ranges;
    
    hid_t file = -1, dset = -1;
    
    // process 0 reads the number of ranges and broadcasts
    if (rank == 0)
      {
        file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(file >= 0);
        dset = H5Dopen2(file, h5types_path_join(POPS).c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        num_ranges = (uint64_t) H5Sget_simple_extent_npoints(fspace);
        assert(num_ranges > 0);
        assert(H5Sclose(fspace) >= 0);
      }
    
    assert(MPI_Bcast(&num_ranges, 1, MPI_UINT64_T, 0, comm) >= 0);
    
    // allocate buffers
    pop_vector.resize(num_ranges);
    
    // MPI rank 0 reads and broadcasts the population ranges
    
    if (rank == 0)
      {
        hid_t ftype = H5Dget_type(dset);
        assert(ftype >= 0);
        hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
        
        assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &pop_vector[0]) >= 0);
        
        assert(H5Tclose(mtype) >= 0);
        assert(H5Tclose(ftype) >= 0);
        
        assert(H5Dclose(dset) >= 0);
        assert(H5Fclose(file) >= 0);
      }
    
    assert(MPI_Bcast(&pop_vector[0], (int)num_ranges*sizeof(pop_range_t),
                     MPI_BYTE, 0, comm) >= 0);
    
    n_nodes = 0;
    for(size_t i = 0; i < pop_vector.size(); ++i)
      {
        pop_ranges.insert(make_pair(pop_vector[i].start,
                                    make_pair(pop_vector[i].count,
                                              pop_vector[i].pop)));
        n_nodes = n_nodes + pop_vector[i].count;
      }
    
    return ierr;
  }
}

