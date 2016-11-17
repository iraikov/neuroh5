#ifndef DBS_READ_TEMPLATE
#define DBS_READ_TEMPLATE

#include "hdf5.h"

#include <cassert>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      template<class T>
      herr_t dbs_read
      (
       hid_t              file,
       const std::string& name,
       const hsize_t&     start,
       const hsize_t&     block,
       hid_t              ntype,
       std::vector<T>&    v
       )
      {
        herr_t ierr = 0;
        hid_t mspace = H5Screate_simple(1, &block, NULL);
        assert(mspace >= 0);
        ierr = H5Sselect_all(mspace);
        assert(ierr >= 0);

        hid_t dset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
        assert(dset >= 0);

        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hsize_t one = 1;
        ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one,
                                   &block);
        assert(ierr >= 0);

        ierr = H5Dread(dset, ntype, mspace, fspace, H5P_DEFAULT, &v[0]);
        assert(ierr >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);

        return ierr;
      }
    }
  }
}

#endif
