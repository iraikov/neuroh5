
#include "hdf5.h"
#include <string>
#include <vector>

#include "exists_group.hh"

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
      
    /*****************************************************************************
     * Check if group exists
     *****************************************************************************/
    int exists_group
    (
     hid_t  file,
     const string& path
     )
    {
      int result=0;
      int status=0;
      H5L_info_t infobuf;
      
      /* Save old error handler */
      H5E_auto_t error_handler;
      void *client_data;
      H5Eget_auto(H5E_DEFAULT, &error_handler, &client_data);

      /* Turn off error handling */
      H5Eset_auto(H5E_DEFAULT, NULL, NULL);

      result = H5Lget_info (file, path.c_str(), &infobuf, H5P_DEFAULT);

      status = H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
      
      return result >= 0;
    }
  }
}
