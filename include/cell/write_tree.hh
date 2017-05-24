#ifndef WRITE_TREE_HH
#define WRITE_TREE_HH

#include "hdf5.h"

#include <vector>

namespace neurotrees
{

/*****************************************************************************
 * Save tree data structures to HDF5
 *****************************************************************************/
  int write_trees
  (
   MPI_Comm comm,
   const std::string& file_name,
   const std::string& pop_name,
   const hsize_t ptr_start,
   const hsize_t attr_start,
   const hsize_t sec_start,
   const hsize_t topo_start,
   std::vector<neurotree_t> &tree_list
   );
}

#endif
