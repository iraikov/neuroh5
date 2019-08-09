#ifndef EXISTS_GROUP_HH
#define EXISTS_GROUP_HH

#include "hdf5.h"

#include <string>

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Checks if a group exists
     *****************************************************************************/
    
    int exists_group
    (
     hid_t  loc,
     const std::string& path
     );
  }
}

#endif
