
#include "exists_dataset.hh"
#include "dataset_num_elements.hh"

#include <throw_assert.hh>

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    hsize_t dataset_num_elements
    (
     const hid_t&  loc,
     const string& path
     )
    {
      hsize_t result = 0;
      herr_t ierr = 0;

      if (exists_dataset (loc, path) > 0)
        {
          hid_t dset = H5Dopen2(loc, path.c_str(), H5P_DEFAULT);
          throw_assert(dset >= 0, "dataset_num_elements: unable to open dataset "  << path);
          hid_t fspace = H5Dget_space(dset);
          throw_assert(fspace >= 0, "dataset_num_elements: unable to get space of dataset " << path);
          result = (hsize_t) H5Sget_simple_extent_npoints(fspace);
          ierr = H5Sclose(fspace);
          throw_assert(ierr >= 0, "dataset_num_elements: unable to close file space");
          ierr = H5Dclose(dset);
          throw_assert(ierr >= 0, "dataset_num_elements: unable to close dataset " << path);
        }
      
      return result;
    }
  }
}
