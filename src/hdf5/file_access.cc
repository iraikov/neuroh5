
#include "hdf5.h"

#include <cassert>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace hdf5
  {

    /*****************************************************************************
     * Routines for opening and closing NeuroH5 for reading and writing.
     *****************************************************************************/
    hid_t open_file
    (
     MPI_Comm comm,
     const std::string& file_name,
     const bool collective = false,
     const bool rdwr = false
     )
    {
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);

      if (collective)
        {
          assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
        }
      
      hid_t file;

      if (rdwr)
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
        }
      else
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        }
      
      assert(file >= 0);
      
      return file;
    }

    herr_t close_file
    (
     hid_t& file
     )
    {
      herr_t status = H5Fclose(file);
      return status;
    }
        
    
  }
}

