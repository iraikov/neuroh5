// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#include "src_index.hh"

#include "debug.hh"
#include "ngh5paths.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{

   herr_t source_index
  (
   MPI_Comm            comm,
   hid_t               file,
   const string&       proj_name,
   const hsize_t&      in_block,
   const DST_PTR_T&     dst_rebase,
   const vector<DST_PTR_T>&  dst_ptr,
   vector<NODE_IDX_T>& src_idx
   )
  {
    herr_t ierr = 0;

    int rank, size;
    assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

    if (in_block > 0)
      {
        DEBUG("Task ",rank,": ", "src_idx: dst_ptr.front() = ", dst_ptr.front(),
              "\n");
        DEBUG("Task ",rank,": ", "src_idx: dst_ptr.back() = ", dst_ptr.back(),
              "\n");

        hsize_t block = (hsize_t)(dst_ptr.back() - dst_ptr.front());
        hsize_t start = (hsize_t)dst_rebase;

        DEBUG("Task ",rank,": ", "src_idx: block = ", block, "\n");
        DEBUG("Task ",rank,": ", "src_idx: start = ", start, "\n");

        if (block > 0)
          {
            // allocate buffer and memory dataspace
            src_idx.resize(block);
            assert(src_idx.size() > 0);

            hid_t mspace = H5Screate_simple(1, &block, NULL);
            assert(mspace >= 0);
            ierr = H5Sselect_all(mspace);
            assert(ierr >= 0);

            hid_t dset = H5Dopen2(file,
                                  ngh5_prj_path(proj_name,
                                                H5PathNames::SRC_IDX).c_str(),
                                  H5P_DEFAULT);
            assert(dset >= 0);

            // make hyperslab selection
            hid_t fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            hsize_t one = 1;
            ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                       &one, &block);
            assert(ierr >= 0);

            ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                           H5P_DEFAULT, &src_idx[0]);
            assert(ierr >= 0);

            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);
          }

        DEBUG("Task ",rank,": ", "src_idx: done\n");
      }

    return ierr;
  }

}
