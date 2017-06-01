#ifndef READ_TREE_HH
#define READ_TREE_HH

#include <hdf5.h>
#include <mpi.h>

#include <vector>

namespace neuroh5
{

  namespace cell
  {
    /*****************************************************************************
     * Load tree data structures from HDF5
     *****************************************************************************/
    
    int read_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T pop_start,
     std::vector<neurotree_t> &tree_list,
     size_t offset = 0,
     size_t numitems = 0,
     bool collective = true
     );
  }

}

#endif
