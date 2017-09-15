
#include "num_projection_blocks.hh"
#include "dataset_num_elements.hh"
#include "path_names.hh"

#include <cassert>

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    
    // Returns the number of blocks in a projection
    hsize_t num_projection_blocks
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      hsize_t num_blocks = 0;

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          assert(file >= 0);
          
          // determine number of blocks in projection
          num_blocks = hdf5::dataset_num_elements
            (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR)) - 1;
        }

      assert(MPI_Bcast(&num_blocks, 1, MPI_UINT64_T, 0, comm) == MPI_SUCCESS);
      
      return num_blocks;
    }

  }
}
