
#include <hdf5.h>
#include <string>

#include "path_names.hh"
#include "exists_tree_dataset.hh"

namespace neuroh5
{
  namespace hdf5
  {
    /*****************************************************************************
     * Check if dataset for tree structures exists
     *****************************************************************************/
    int exists_tree_dataset
    (
     hid_t  file,
     const std::string& pop_name
     )
    {
      herr_t status;  
      
      status = H5Lexists (file, POPULATIONS.c_str(), H5P_DEFAULT);
      if (status)
        {
            status = H5Lexists (file, population_path(pop_name).c_str(), H5P_DEFAULT);
        }

      return status;
    }
  }
}
