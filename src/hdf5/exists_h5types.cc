
#include "hdf5.h"
#include <string>

#include "hdf5_path_names.hh"
#include "exists_tree_h5types.hh"

namespace neuroio
{
  namespace io
  {
    namespace hdf5
    {
      
      /*****************************************************************************
       * Check if dataset for type definitions exists
       *****************************************************************************/
      int hdf5_exists_tree_h5types
      (
       hid_t  file
       )
      {
        herr_t status;  

        status = H5Lexists (file, H5_TYPES.c_str(), H5P_DEFAULT);
        if (status)
          {
            status = H5Lexists (file, h5types_path_join(POPS).c_str(), H5P_DEFAULT);
          }

        return status;
      }
    }
  }
}
