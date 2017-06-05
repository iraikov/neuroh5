
#include "dataset_num_elements.hh"

#include <cassert>

using namespace std;

namespace ngh5
{
  namespace hdf5
  {
    hsize_t dataset_num_elements
    (
     MPI_Comm      comm,
     const hid_t&  loc,
     const string& path
     )
    {
      hsize_t result = 0;
      
      hid_t dset = H5Dopen2(loc, path.c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      result = (hsize_t) H5Sget_simple_extent_npoints(fspace);
      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
      
      return result;
    }
  }
}
