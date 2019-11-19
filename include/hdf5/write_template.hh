#ifndef WRITE_TEMPLATE
#define WRITE_TEMPLATE

#include "hdf5.h"

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
        throw_assert(dset >= 0,
                     "write: unable to open dataset " << name);

        if (newsize > 0)
          {
            ierr = H5Dset_extent (dset, &newsize);
            throw_assert(ierr >= 0,
                         "write: unable to set extent on dataset "
                         << name << " to " << newsize);
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
            throw_assert(ierr >= 0,
                         "write: unable to select hyperslab "
                         << start << ":" << len << " from dataset " << name);
            ierr = H5Sselect_all(mspace);
            throw_assert(ierr >= 0,
                         "write: error in H5Sselect_all "
                          << " on dataset " << name);

          }
        else
          {
            ierr = H5Sselect_none(fspace);
            throw_assert(ierr >= 0,
                         "write: error in H5Sselect_none"
                          << " on dataset " << name);
            ierr = H5Sselect_none(mspace);
            throw_assert(ierr >= 0,
                         "write: error in H5Sselect_none"
                          << " on dataset " << name);
          }

        ierr = H5Dwrite(dset, ntype, mspace, fspace, wapl, &v[0]);
	if (ierr < 0)
	  {
	    H5Eprint2(H5E_DEFAULT, stdout);
	  }
        throw_assert(ierr >= 0, "write_template: error in H5Dwrite on dataset "
                     << name << " length: " << len);

        throw_assert_nomsg(H5Dclose(dset) >= 0);
        throw_assert_nomsg(H5Sclose(mspace) >= 0);
        throw_assert_nomsg(H5Sclose(fspace) >= 0);

        return ierr;
      }


  }
}

#endif
