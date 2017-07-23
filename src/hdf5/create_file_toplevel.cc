
#include "hdf5.h"

#include <cassert>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace hdf5
  {

    /*****************************************************************************
     * Creates a file with the specified top-level groups for storing NeuroH5 structures
     * and returns operation status
     *****************************************************************************/
    int create_file_toplevel
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::vector<std::string>& groups
     )
    {
      herr_t status;  
      hid_t file, group, prop;
      
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
      
      /* Create a new file. If file exists its contents will be overwritten. */
      file = H5Fcreate (file_name.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fapl);
      assert(file >= 0);
      
      /* Create dataset creation properties, i.e. to enable chunking  */
      prop = H5Pcreate (H5P_DATASET_CREATE);
    
      /* Creates the specified groups.  */
      for (size_t i=0; i<groups.size(); i++)
        {
          group = H5Gcreate2(file, groups[i].c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Gclose(group);
        }
      
      status = H5Pclose (prop);
      status = H5Pclose (fapl);
      status = H5Fclose(file);
      
      return status;
    }
  }
}

