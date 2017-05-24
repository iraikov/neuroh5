
#include "hdf5.h"
#include <string>

#include "hdf5_path_names.hh"
#include "hdf5_exists_tree_dataset.hh"

namespace neurotrees
{

/*****************************************************************************
 * Check if dataset for tree structures exists
 *****************************************************************************/
int hdf5_exists_tree_dataset
  (
   hid_t  file,
   const std::string& pop_name
   )
  {
    herr_t status;  

    status = H5Lexists (file, POPS.c_str(), H5P_DEFAULT);
    if (status)
      {
        status = H5Lexists (file, population_path(pop_name).c_str(), H5P_DEFAULT);
      }

    return status;
  }
}
