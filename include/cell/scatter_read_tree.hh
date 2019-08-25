#ifndef SCATTER_READ_TREE_HH
#define SCATTER_READ_TREE_HH

#include <hdf5.h>
#include <mpi.h>

#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"

namespace neuroh5
{

  namespace cell
  {

    /*****************************************************************************
     * Load tree data structures from HDF5 and scatter to all ranks
     *****************************************************************************/
    
    int scatter_read_trees
    ( 
     MPI_Comm                              all_comm,
     const std::string&                    file_name,
     const int                             io_size,
     const std::vector<std::string>       &attr_name_spaces,
     // A vector that maps nodes to compute ranks
     const std::map<CELL_IDX_T, rank_t>&    node_rank_map,
     const string                         &pop_name,
     const CELL_IDX_T                      pop_start,
     std::map<CELL_IDX_T, neurotree_t>    &tree_map,
     std::map<string, data::NamedAttrMap> &attr_maps,
     size_t offset,
     size_t numitems
     );
  }
}

#endif
