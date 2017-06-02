#ifndef EXISTS_TREE_DATASET_HH
#define EXISTS_TREE_DATASET_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    /*****************************************************************************
     * Checks if a tree dataset exists for the given population
     *****************************************************************************/
    
    int exists_tree_dataset
    (
     hid_t  file,
     const std::string& pop_name
     );
  }
}

#endif
