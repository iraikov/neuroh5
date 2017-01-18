// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file write_dbs_projection.cc
///
///  Functions for writing edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "ngh5_types.hh"
#include "write_dbs_projection.hh"

#include "dataset_num_elements.hh"
#include "debug.hh"
#include "hdf5_write_template.hh"
#include "hdf5_path_names.hh"
#include "sort_permutation.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>


namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      
      
      herr_t write_dbs_projection
      (
       MPI_Comm                   comm,
       const std::string&         file_name,
       const std::string&         proj_name,
       const NODE_IDX_T&          dst_start,
       const NODE_IDX_T&          src_start,
       const DST_BLK_PTR_T&       block_base,
       const DST_PTR_T&           edge_base,
       const vector<NODE_IDX_T>&  dst_idx,
       const vector<NODE_IDX_T>&  src_idx
       )
      {
        

        // get a file handle and retrieve the MPI info
        hid_t file = H5Iget_file_id(loc);
        assert(file >= 0);

        MPI_Comm comm;
        MPI_Info info;
        hid_t fapl = H5Fget_access_plist(file);
        assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        uint32_t my_count = (uint32_t)edge_id.size();
        std::vector<uint32_t> all_counts(size);
        assert(MPI_Allgather(&my_count, 1, MPI_UINT32_T, &all_counts[0], 1,
                             MPI_UINT32_T, comm) == MPI_SUCCESS);

        // calculate the total dataset size and the offset of my piece
        hsize_t start = 0, total = 0, count = 1, block = my_count;
        for (size_t p = 0; p < size; ++p)
          {
            if (p < rank)
              {
                start += (hsize_t) all_counts[p];
              }
            total += (hsize_t) all_counts[p];
          }

        // create dataspaces and selections
        hid_t mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        hid_t fspace = H5Screate_simple(1, &total, &total);
        assert(fspace >= 0);
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &count, &block) >= 0);

        // we write the values first
        // everything needs to be scaled by 2 for the edge IDs

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        hid_t dset = H5Dcreate(loc, (path + "/value").c_str(), ftype, fspace,
                               lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, mtype, mspace, fspace, H5P_DEFAULT, &value[0])
               >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Tclose(mtype) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);

        // scale by factor 2
        block *= 2;
        mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        total *= 2;
        fspace = H5Screate_simple(1, &total, &total);
        assert(fspace >= 0);
        start *= 2;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &count, &block) >= 0);

        dset = H5Dcreate(loc, (path + "/edge_id").c_str(), H5T_STD_U32LE,
                         fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, H5T_NATIVE_UINT32, mspace, fspace, H5P_DEFAULT,
                        &edge_id[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Pclose(lcpl) >= 0);
      }
    }
  }
}
