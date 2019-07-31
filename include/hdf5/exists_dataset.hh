#ifndef EXISTS_DATASET_HH
#define EXISTS_DATASET_HH

#include "hdf5.h"

#include <string>

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Checks if a dataset exist
     *****************************************************************************/
    
    int exists_dataset
    (
     hid_t  loc,
     const std::string& path
     );
  }
}

#endif
