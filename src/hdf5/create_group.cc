#include "hdf5.h"

#include <vector>
#include <string>
#include <cassert>

#include "exists_dataset.hh"
using namespace std;

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Creates an HDF5 group
     *****************************************************************************/

    herr_t create_group
    (
     const hid_t&    file,
     const string&   path
     )
    {
      herr_t status = 0;
      if (!(exists_dataset (file, path) > 0))
        {
        
          hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
          assert(lcpl >= 0);
          assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
        
          hid_t group = H5Gcreate2(file, path.c_str(),
                                   lcpl, H5P_DEFAULT, H5P_DEFAULT);
          assert(group >= 0);
          status = H5Gclose(group);
          assert(status == 0);
          status = H5Pclose(lcpl);
          assert(status == 0);
        }
    
      return status;
    }
  }

}
