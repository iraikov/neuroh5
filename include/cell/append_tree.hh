#ifndef APPEND_TREE_HH
#define APPEND_TREE_HH

#include <hdf5.h>

#include <forward_list>

#include "neuroh5_types.hh"
#include "cell_index.hh"
#include "cell_attributes.hh"
#include "compact_optional.hh"
#include "optional_value.hh"
#include "throw_assert.hh"

namespace neuroh5
{
  namespace cell
  {

    int build_tree_datasets
    (
     MPI_Comm comm,

     std::forward_list<neurotree_t> &tree_list,

     std::vector<SEC_PTR_T>& sec_ptr,
     std::vector<TOPO_PTR_T>& topo_ptr,
     std::vector<ATTR_PTR_T>& attr_ptr,
    
     std::vector<CELL_IDX_T>& all_index_vector,
     std::vector<SECTION_IDX_T>& all_src_vector,
     std::vector<SECTION_IDX_T>& all_dst_vector,
     std::vector<COORD_T>& all_xcoords,
     std::vector<COORD_T>& all_ycoords,
     std::vector<COORD_T>& all_zcoords,  // coordinates of nodes
     std::vector<REALVAL_T>& all_radiuses,    // Radius
     std::vector<LAYER_IDX_T>& all_layers,        // Layer
     std::vector<SECTION_IDX_T>& all_sections,    // Section
     std::vector<PARENT_NODE_IDX_T>& all_parents, // Parent
     std::vector<SWC_TYPE_T>& all_swc_types // SWC Types
     );
    
    int build_singleton_tree_datasets
    (
     MPI_Comm comm,

     std::forward_list<neurotree_t> &tree_list,
    
     std::vector<SECTION_IDX_T>& all_src_vector,
     std::vector<SECTION_IDX_T>& all_dst_vector,
     std::vector<COORD_T>& all_xcoords,
     std::vector<COORD_T>& all_ycoords,
     std::vector<COORD_T>& all_zcoords,  // coordinates of nodes
     std::vector<REALVAL_T>& all_radiuses,    // Radius
     std::vector<LAYER_IDX_T>& all_layers,        // Layer
     std::vector<SECTION_IDX_T>& all_sections,    // Section
     std::vector<PARENT_NODE_IDX_T>& all_parents, // Parent
     std::vector<SWC_TYPE_T>& all_swc_types // SWC Types
     );

    
    /*****************************************************************************
     * Save tree data structures to HDF5
     *****************************************************************************/
    int append_trees
    (
     MPI_Comm comm,
     MPI_Comm io_comm,
     hid_t loc,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::forward_list<neurotree_t> &tree_list,
     const set<size_t>              &io_rank_set,
     CellPtr ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000
     );

    int append_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::forward_list<neurotree_t> &tree_list,
     size_t io_size,
     const size_t chunk_size,
     const size_t value_chunk_size
     );

  }
}

#endif
