#ifndef WRITE_TEMPLATE
#define WRITE_TEMPLATE

#include "hdf5.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "throw_assert.hh"

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
       const hid_t        wapl
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
        throw_assert(fspace >= 0, "error in H5Dget_space");
        hsize_t one = 1;
        hid_t mspace = H5Screate_simple(1, &len, NULL);
        throw_assert(mspace >= 0, "error in H5Screate_simple");

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

        ierr = H5Dwrite(dset, ntype, mspace, fspace, wapl, &v[0]);
	if (ierr < 0)
	  {
	    H5Eprint2(H5E_DEFAULT, stdout);
	  }
        throw_assert(ierr >= 0, "write_template: error in H5Dwrite");

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        return ierr;
      }


  }
}

#endif
