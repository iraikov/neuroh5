#ifndef HDF5_CREATE_TREE_FILE_HH
#define HDF5_CREATE_TREE_FILE_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Creates an empty tree file
 *****************************************************************************/

  int hdf5_create_tree_file
  (
   MPI_Comm comm,
   const std::string& file_name
   );
}

#endif
