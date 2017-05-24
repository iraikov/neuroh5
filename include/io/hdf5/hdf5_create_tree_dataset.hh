#ifndef HDF5_CREATE_TREE_DATASET_HH
#define HDF5_CREATE_TREE_DATASET_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Creates an empty tree dataset
 *****************************************************************************/

  int hdf5_create_tree_dataset
  (
   MPI_Comm comm,
   hid_t  file,
   const std::string& pop_name
   );
}

#endif
