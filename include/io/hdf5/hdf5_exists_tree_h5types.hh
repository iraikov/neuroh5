#ifndef HDF5_EXISTS_TREE_H5TYPES_HH
#define HDF5_EXISTS_TREE_H5TYPES_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Checks if the population definitions for tree dataset exist
 *****************************************************************************/

  int hdf5_exists_tree_h5types
  (
   hid_t  file
   );
}

#endif
