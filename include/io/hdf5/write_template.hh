#ifndef WRITE_TEMPLATE
#define WRITE_TEMPLATE

#include "hdf5.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      template<class T>
      herr_t write
      (
       hid_t&                file,
       const std::string&    dset_name,
       const hid_t&          file_type,
       const std::vector<T>& v,
       hid_t                 dcpl = H5P_DEFAULT,
       bool                  do_coll_io = false
       )
      {
        herr_t ierr = 0;

        // get the I/O communicator
        MPI_Comm comm;
        MPI_Info info;
        hid_t fapl = H5Fget_access_plist(file);
        assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        assert(H5Pclose(fapl) >= 0);

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        hid_t dxpl = H5P_DEFAULT;
        if (do_coll_io)
          {
            dxpl = H5Pcreate(H5P_DATASET_XFER);
            assert(dxpl >= 0);
            assert(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE) >= 0);
          }

        // determine the total size of the dataset and create it
        hsize_t my_size = v.size(), total_size = 0;
        assert(MPI_Allreduce(&my_size, &total_size, 1, MPI_UINT64_T, MPI_SUM,
                             comm) == MPI_SUCCESS);
        hid_t fspace = H5Screate_simple(1, &total_size, &total_size);
        assert(fspace >= 0);
        hid_t dset = H5Dcreate(file, dset_name.c_str(), file_type, fspace,
                               lcpl, dcpl, H5P_DEFAULT);
        assert(dset >= 0);

        // prepare the memory spaces and hyperslab selections
        // if my rank doesn't have to write anything, we fake it
        hsize_t one = 1, start, block = (hsize_t)my_size;
        if (my_size == 0)
          {
            block = 1;
          }
        hid_t mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        if (my_size > 0)
          {
            assert(H5Sselect_all(mspace) >= 0);
          }
        else
          {
            assert(H5Sselect_none(mspace) >= 0);
          }

        // determine the start
        std::vector<uint64_t> sendbuf(size, my_size), recvbuf(size);
        assert(MPI_Allgather(&sendbuf[0], 1, MPI_UINT64_T, &recvbuf[0], 1,
                             MPI_UINT64_T, comm) == MPI_SUCCESS);
        for (int p = 0; p < rank; ++p)
          {
            start += (hsize_t) recvbuf[p];
          }
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one,
                                   &block) >= 0);

        // infer the memory type
        hid_t mtype = H5Tget_native_type(file_type, H5T_DIR_ASCEND);
        assert(mtype >= 0);

        // Go!
        assert(H5Dwrite(dset, mtype, mspace, fspace, dxpl, &v[0]) >= 0);

        assert(H5Tclose(mtype) >= 0);
        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(fspace) >= 0);
        if (dxpl != H5P_DEFAULT)
          {
            assert(H5Pclose(dxpl) >= 0);
          }
        assert(H5Pclose(lcpl) >= 0);

        return ierr;
      }
    }
  }
}

#endif
