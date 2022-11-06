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
    template <class T>
    int append_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::forward_list<neurotree_t> &tree_list,
     CellPtr ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000
     )
    {
      herr_t status; 

      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      
      std::vector<SEC_PTR_T> sec_ptr;
      std::vector<TOPO_PTR_T> topo_ptr;
      std::vector<ATTR_PTR_T> attr_ptr;
    
      std::vector<CELL_IDX_T> all_index_vector;
      std::vector<SECTION_IDX_T> all_src_vector, all_dst_vector;
      std::vector<COORD_T> all_xcoords, all_ycoords, all_zcoords;  // coordinates of nodes
      std::vector<REALVAL_T> all_radiuses;    // Radius
      std::vector<LAYER_IDX_T> all_layers;        // Layer
      std::vector<SECTION_IDX_T> all_sections;    // Section
      std::vector<PARENT_NODE_IDX_T> all_parents; // Parent
      std::vector<SWC_TYPE_T> all_swc_types; // SWC Types

      if (ptr_type.type == PtrNone)
        {
          status = build_singleton_tree_datasets(comm,
                                                 tree_list,
                                                 all_src_vector, all_dst_vector,
                                                 all_xcoords, all_ycoords, all_zcoords, 
                                                 all_radiuses, all_layers, all_sections,
                                                 all_parents, all_swc_types);
        }
      else
        {
          status = build_tree_datasets(comm,
                                       tree_list,
                                       sec_ptr, topo_ptr, attr_ptr,
                                       all_index_vector, all_src_vector, all_dst_vector,
                                       all_xcoords, all_ycoords, all_zcoords, 
                                       all_radiuses, all_layers, all_sections,
                                       all_parents, all_swc_types);
          throw_assert_nomsg(status >= 0);
        }

            
      throw_assert_nomsg(append_cell_index (comm, file_name, pop_name, pop_start,
                                            hdf5::TREES, all_index_vector) == 0);

      const data::optional_hid dflt_data_type;
      const data::optional_hid coord_data_type(COORD_H5_NATIVE_T);
      const data::optional_hid layer_data_type(LAYER_IDX_H5_NATIVE_T);
      const data::optional_hid parent_node_data_type(PARENT_NODE_IDX_H5_NATIVE_T);
      const data::optional_hid section_data_type(SECTION_IDX_H5_NATIVE_T);
      const data::optional_hid swc_data_type(SWC_TYPE_H5_NATIVE_T);

      string attr_ptr_owner_path = hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::X_COORD) + "/" + hdf5::ATTR_PTR;
      string sec_ptr_owner_path  = hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SRCSEC) + "/" + hdf5::SEC_PTR;

      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::X_COORD,
                             all_index_vector, attr_ptr, all_xcoords,
                             coord_data_type, IndexShared,
                             CellPtr (PtrOwner, hdf5::ATTR_PTR),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::Y_COORD,
                             all_index_vector, attr_ptr, all_ycoords,
                             coord_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::Z_COORD,
                             all_index_vector, attr_ptr, all_zcoords,
                             coord_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::RADIUS,
                             all_index_vector, attr_ptr, all_radiuses,
                             dflt_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::LAYER,
                             all_index_vector, attr_ptr, all_layers,
                             layer_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::PARENT,
                             all_index_vector, attr_ptr, all_parents,
                             parent_node_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);

      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::SWCTYPE,
                             all_index_vector, attr_ptr, all_swc_types,
                             swc_data_type, IndexShared,
                             CellPtr (PtrShared, attr_ptr_owner_path),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::SRCSEC,
                             all_index_vector, topo_ptr, all_src_vector,
                             section_data_type, IndexShared,
                             CellPtr (PtrOwner, hdf5::SEC_PTR),
                             chunk_size, value_chunk_size);
      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::DSTSEC,
                             all_index_vector, topo_ptr, all_dst_vector,
                             section_data_type, IndexShared,
                             CellPtr (PtrShared, sec_ptr_owner_path),
                             chunk_size, value_chunk_size);

      append_cell_attribute (comm, file_name, hdf5::TREES, pop_name, pop_start, hdf5::SECTION,
                             all_index_vector, sec_ptr, all_sections,
                             section_data_type, IndexShared,
                             CellPtr (PtrOwner, hdf5::SEC_PTR),
                             chunk_size, value_chunk_size);

        
      return 0;
    }

    template<class T>
    int append_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::forward_list<neurotree_t> &tree_list,
     const size_t chunk_size,
     const size_t value_chunk_size
     )
    {
      return append_trees<T>(comm, file_name, pop_name, pop_start, tree_list,
                             CellPtr(PtrOwner), chunk_size, value_chunk_size);
    }

  }
}

#endif
