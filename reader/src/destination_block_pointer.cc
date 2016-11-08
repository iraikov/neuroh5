// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#include "destination_block_pointer.hh"

#include "debug.hh"
#include "ngh5paths.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{

  herr_t destination_block_pointer
  (
   MPI_Comm               comm,
   hid_t                  file,
   const string&          proj_name,
   const hsize_t&         start,
   const hsize_t&         block,
   vector<DST_BLK_PTR_T>& dst_blk_ptr,
   DST_BLK_PTR_T&         block_rebase
   )
  {
    herr_t ierr = 0;

    int rank, size;
    assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

    if (block > 0)
      {
        // allocate buffer and memory dataspace
        dst_blk_ptr.resize(block);

        hid_t mspace = H5Screate_simple(1, &block, NULL);
        assert(mspace >= 0);
        ierr = H5Sselect_all(mspace);
        assert(ierr >= 0);

        cerr << flush;

        hid_t dset = H5Dopen2(file,
                              ngh5_prj_path(proj_name,
                                            H5PathNames::DST_BLK_PTR).c_str(),
                              H5P_DEFAULT);
        assert(dset >= 0);

        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hsize_t one = 1;
        ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one,
                                   &block);
        assert(ierr >= 0);

        ierr = H5Dread(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                       H5P_DEFAULT, &dst_blk_ptr[0]);
        assert(ierr >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);

        // rebase the block_ptr array to local offsets
        // REBASE is going to be the start offset for the hyperslab

        block_rebase = dst_blk_ptr[0];
        DEBUG("Task ",rank,": ","block_rebase = ", block_rebase, "\n");

        for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
          {
            dst_blk_ptr[i] -= block_rebase;
          }
      }

    return ierr;
  }

}
