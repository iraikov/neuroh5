
#include <mpi.h>
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
      // MPI group color value used for I/O rank 0
      int io_color = 1, color;
      
      int rank, size;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "create_file_toplevel: error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "create_file_toplevel: error in MPI_Comm_rank");
      
      // MPI Communicator for I/O rank 0
      MPI_Comm comm_0;

      if (rank == 0) {
          color = io_color;
        } else
          {
            color = 0;
          }
      MPI_Comm_split(comm,color,rank,&comm_0);
      MPI_Comm_set_errhandler(comm_0, MPI_ERRORS_RETURN);

      if (rank == 0) { 
      
	  /* Create a new file. If file exists its contents will be overwritten. */
	  file = H5Fcreate (file_name.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
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
	  status = H5Fclose(file);
	  throw_assert(status >= 0, "create_file_toplevel: unable to close file " << file_name);
      }

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
      throw_assert(MPI_Comm_free(&comm_0) == MPI_SUCCESS,
		   "create_file_toplevel: error in MPI_Comm_free");

      
      return status;
    }
  }
}

