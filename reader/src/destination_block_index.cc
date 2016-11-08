// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#include "destination_block_index.hh"

#include "debug.hh"
#include "ngh5paths.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{

  herr_t destination_block_index
  (
   MPI_Comm            comm,
   hid_t               file,
   const string&       proj_name,
   const hsize_t&      start,
   const hsize_t&      in_block,
   vector<NODE_IDX_T>& dst_blk_idx
   )
  {
    herr_t ierr = 0;

    int rank, size;
    assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

    if (in_block > 0)
      {
        hsize_t block = in_block;

        if (rank == size-1)
          {
            block = in_block-1;
          }

        dst_blk_idx.resize(block);
        assert(dst_blk_idx.size() > 0);

        DEBUG("Task ",rank,": ", "dst_idx: block = ", block, "\n");
        DEBUG("Task ",rank,": ", "dst_idx: start = ", start, "\n");

        hid_t mspace = H5Screate_simple(1, &block, NULL);
        assert(mspace >= 0);
        ierr = H5Sselect_all(mspace);
        assert(ierr >= 0);

        hid_t dset = H5Dopen2(file,
                              ngh5_prj_path(proj_name,
                                            H5PathNames::DST_BLK_IDX).c_str(),
                              H5P_DEFAULT);
        assert(dset >= 0);

        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hsize_t one = 1;
        ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one,
                                   &block);
        assert(ierr >= 0);

        ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT,
                       &dst_blk_idx[0]);
        assert(ierr >= 0);

        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
      }

    return ierr;
  }

}
