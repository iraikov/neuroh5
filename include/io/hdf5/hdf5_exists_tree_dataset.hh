#ifndef HDF5_EXISTS_TREE_DATASET_HH
#define HDF5_EXISTS_TREE_DATASET_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Checks if a tree dataset exists for the given population
 *****************************************************************************/

  int hdf5_exists_tree_dataset
  (
   hid_t  file,
   const std::string& pop_name
   );
}

#endif
