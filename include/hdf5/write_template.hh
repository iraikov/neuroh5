#ifndef WRITE_TEMPLATE
#define WRITE_TEMPLATE

#include "hdf5.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    template<class T>
      herr_t write
      (
       const hid_t&       loc,
       const std::string& name,
       const hsize_t&     newsize,
       const hsize_t&     start,
       const hsize_t&     len,
       const hid_t&       ntype,
       const std::vector<T>&    v,
       const bool do_coll_io = true
       )
      {

        herr_t ierr = 0;

        hid_t dset = H5Dopen2(loc, name.c_str(), H5P_DEFAULT);
        assert(dset >= 0);

        if (newsize > 0)
          {
            ierr = H5Dset_extent (dset, &newsize);
            assert(ierr >= 0);
          }

        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hsize_t one = 1;
        hid_t mspace = H5Screate_simple(1, &len, NULL);
        assert(mspace >= 0);

        if (len > 0)
          {
            ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &len);
            assert(ierr >= 0);
            ierr = H5Sselect_all(mspace);
            assert(ierr >= 0);
          }
        else
          {
            ierr = H5Sselect_none(fspace);
            assert(ierr >= 0);
            ierr = H5Sselect_none(mspace);
            assert(ierr >= 0);
          }

        hid_t dxpl = H5P_DEFAULT;
        if (do_coll_io)
          {
            dxpl = H5Pcreate(H5P_DATASET_XFER);
            assert(dxpl >= 0);
            assert(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE) >= 0);
          }

        ierr = H5Dwrite(dset, ntype, mspace, fspace, dxpl, &v[0]);
        assert(ierr >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        return ierr;
      }


  }
}

#endif
