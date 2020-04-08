
#include <hdf5.h>
#include <string>

#include "dataset_type.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    hid_t dataset_type
    (
     hid_t file,
     const string& path
     )
    {
      herr_t ierr = 0;
      hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
      throw_assert_nomsg(dset >= 0);

      hid_t ftype = H5Dget_type(dset);
      throw_assert_nomsg(ftype >= 0);
      hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

      ierr = H5Dclose(dset);
      throw_assert_nomsg(ierr >= 0);
    
      return ntype;
    }
  }
}
