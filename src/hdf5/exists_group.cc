
#include "hdf5.h"
#include <string>
#include <vector>

#include "tokenize.hh"
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
      const string delim = "/";
      herr_t status=-1;  string ppath;
      vector<string> tokens;
      data::tokenize (path, delim, tokens);
      
      for (string value : tokens)
        {
          ppath = ppath + delim + value;
          status = H5Lexists (file, ppath.c_str(), H5P_DEFAULT);
          if (status <= 0)
            break;
        }
      
      return status;
    }
  }
}
