
#include <hdf5.h>

#include <string>
#include <vector>

#include "throw_assert.hh"
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
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "create_file_toplevel: error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "create_file_toplevel: error in MPI_Comm_rank");
      
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert(fapl >= 0, "create_file_toplevel: unable to create file access property list");
#ifdef HDF5_IS_PARALLEL
      status = H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
      throw_assert(status >= 0, "create_file_toplevel: unable to set mpio");
#endif
      
      /* Create a new file. If file exists its contents will be overwritten. */
      file = H5Fcreate (file_name.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fapl);
      throw_assert(file >= 0, "create_file_toplevel: unable to create file " << file_name);
      
      /* Create dataset creation properties, i.e. to enable chunking  */
      prop = H5Pcreate (H5P_DATASET_CREATE);
    
      /* Creates the specified groups.  */
      for (size_t i=0; i<groups.size(); i++)
        {
          group = H5Gcreate2(file, groups[i].c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          throw_assert(group >= 0, "create_file_toplevel: unable to create group " << groups[i]);
          status = H5Gclose(group);
          throw_assert(status >= 0, "create_file_toplevel: unable to close group " << groups[i]);
        }
      
      status = H5Pclose (prop);
      throw_assert(status >= 0, "create_file_toplevel: unable to close property list");
      status = H5Pclose (fapl);
      throw_assert(status >= 0, "create_file_toplevel: unable to close property list");
      status = H5Fclose(file);
      throw_assert(status >= 0, "create_file_toplevel: unable to close file " << file_name);
      
      return status;
    }
  }
}

