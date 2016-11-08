// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#include "destination_pointer.hh"

#include "debug.hh"
#include "ngh5paths.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{

  herr_t destination_pointer
  (
   MPI_Comm                     comm,
   hid_t                        file,
   const string&                proj_name,
   const hsize_t&               in_block,
   const DST_BLK_PTR_T&         block_rebase,
   const vector<DST_BLK_PTR_T>& dst_blk_ptr,
   vector<DST_PTR_T>&           dst_ptr,
   DST_PTR_T&                   edge_base,
   DST_PTR_T&                   dst_rebase
   )
  {
    herr_t ierr = 0;

    int rank, size;
    assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

    // determine my read block of dst_ptr
    if (in_block > 0)
      {
        DEBUG("Task ",rank,": ", "dst_ptr: dst_blk_ptr.front() = ",
              dst_blk_ptr.front(), "\n");
        DEBUG("Task ",rank,": ", "dst_ptr: dst_blk_ptr.back() = ",
              dst_blk_ptr.back(), "\n");

        hsize_t block = in_block;
        if (rank < size-1)
          {
            block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front() + 1);
          }
        else
          {
            block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
          }

        hsize_t start = (hsize_t)block_rebase;
        dst_ptr.resize(block);
        assert(dst_ptr.size() > 0);

        DEBUG("Task ",rank,": ", "dst_ptr: start = ", start, "\n");
        DEBUG("Task ",rank,": ", "dst_ptr: block = ", block, "\n");

        // allocate buffer and memory dataspace

        hid_t mspace = H5Screate_simple(1, &block, NULL);
        assert(mspace >= 0);
        ierr = H5Sselect_all(mspace);
        assert(ierr >= 0);

        hid_t dset = H5Dopen2(file,
                              ngh5_prj_path(proj_name,
                                            H5PathNames::DST_PTR).c_str(),
                              H5P_DEFAULT);
        assert(dset >= 0);

        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hsize_t one = 1;
        ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one,
                                   &block);
        assert(ierr >= 0);

        ierr = H5Dread(dset, DST_PTR_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT,
                       &dst_ptr[0]);
        assert(ierr >= 0);

        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);

        dst_rebase = dst_ptr[0];
        edge_base = dst_rebase;
        DEBUG("Task ",rank,": ", "dst_ptr: dst_rebase = ", dst_rebase, "\n");
        for (size_t i = 0; i < dst_ptr.size(); ++i)
          {
            dst_ptr[i] -= dst_rebase;
          }
      }

    return ierr;
  }

}
