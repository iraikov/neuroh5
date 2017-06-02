#ifndef CREATE_TREE_DATASET_HH
#define CREATE_TREE_DATASET_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{

  namespace hdf5
  {
    /*****************************************************************************
     * Creates an empty tree dataset
     *****************************************************************************/
    
    int create_tree_dataset
    (
     MPI_Comm comm,
     hid_t  file,
     const std::string& pop_name
     );
  }
}

#endif
