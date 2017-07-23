#ifndef EXISTS_TREE_H5TYPES_HH
#define EXISTS_TREE_H5TYPES_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Checks if the population definitions for tree dataset exist
     *****************************************************************************/
    
    int exists_tree_h5types
    (
     hid_t  file
     );
  }
}

#endif
