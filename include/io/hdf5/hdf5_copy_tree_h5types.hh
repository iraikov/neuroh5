#ifndef HDF5_COPY_TREE_H5TYPES_HH
#define HDF5_COPY_TREE_H5TYPES_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Copies population definitions for tree dataset from src file to dst file
 *****************************************************************************/

  int hdf5_copy_tree_h5types
  (
   hid_t  src_file,
   hid_t  dst_file
   );
}

#endif
