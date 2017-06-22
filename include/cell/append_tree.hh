#ifndef APPEND_TREE_HH
#define APPEND_TREE_HH

#include <hdf5.h>

#include <vector>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace cell
  {
    
    /*****************************************************************************
     * Save tree data structures to HDF5
     *****************************************************************************/
    int append_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const hsize_t ptr_start,
     const hsize_t attr_start,
     const hsize_t sec_start,
     const hsize_t topo_start,
     std::vector<neurotree_t> &tree_list,
     CellPtr ptr_type = CellPtr(PtrOwner)
     );
  }
}

#endif
