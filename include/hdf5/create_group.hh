#ifndef CREATE_GROUP_HH
#define CREATE_GROUP_HH

#include "hdf5.h"

#include <vector>

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
     );
    
  }

}

#endif

