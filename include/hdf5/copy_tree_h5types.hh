#ifndef COPY_TREE_H5TYPES_HH
#define COPY_TREE_H5TYPES_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Copies population definitions for tree dataset from src file to dst file
     *****************************************************************************/
    
    int copy_tree_h5types
    (
     hid_t  src_file,
     hid_t  dst_file
     );
  }
}

#endif
