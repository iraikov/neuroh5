// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_tree.cc
///
///  Read tree structures.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>

#include <cassert>
#include <vector>

#include "neuroh5_types.hh"
#include "cell_attributes.hh"
#include "attr_map.hh"

namespace neuroh5
{

  namespace cell
  {

    void append_tree_list (const CELL_IDX_T pop_start,
                           data::NamedAttrMap&       attr_values,
                           std::vector<neurotree_t>& tree_list)
    {
      
      for (CELL_IDX_T idx : attr_values.index_set)
        {
          const vector<SECTION_IDX_T>& src_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::SRCSEC, idx);
          const vector<SECTION_IDX_T>& dst_vector = attr_values.find_name<SECTION_IDX_T>(hdf5::DSTSEC, idx);

          const vector<SECTION_IDX_T>& sections = attr_values.find_name<SECTION_IDX_T>(hdf5::SECTION, idx);
        
          const vector<COORD_T>& xcoords =  attr_values.find_name<COORD_T>(hdf5::X_COORD, idx);
          const vector<COORD_T>& ycoords =  attr_values.find_name<COORD_T>(hdf5::Y_COORD, idx);
          const vector<COORD_T>& zcoords =  attr_values.find_name<COORD_T>(hdf5::Z_COORD, idx);

          const vector<REALVAL_T>& radiuses    =  attr_values.find_name<REALVAL_T>(hdf5::RADIUS, idx);
          const vector<LAYER_IDX_T>& layers    =  attr_values.find_name<LAYER_IDX_T>(hdf5::LAYER, idx);
          const vector<PARENT_NODE_IDX_T>& parents = attr_values.find_name<PARENT_NODE_IDX_T>(hdf5::PARENT, idx);
          const vector<SWC_TYPE_T> swc_types   = attr_values.find_name<SWC_TYPE_T>(hdf5::SWCTYPE, idx);

          CELL_IDX_T gid = pop_start+idx;
          tree_list.push_back(make_tuple(gid,
                                         src_vector, dst_vector, sections,
                                         xcoords,ycoords,zcoords,
                                         radiuses,layers,parents,
                                         swc_types));
        }
    }

    

  
    /*****************************************************************************
     * Load tree data structures from HDF5
     *****************************************************************************/
    int read_trees
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::vector<neurotree_t> &tree_list,
     size_t offset = 0,
     size_t numitems = 0,
     bool collective = true
     )
    {
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      data::NamedAttrMap attr_values;
      
      read_cell_attributes (comm, file_name, hdf5::TREES,
                            pop_name, pop_start, attr_values,
                            offset, numitems);

      append_tree_list (pop_start, attr_values, tree_list);
    
      return 0;
    }


    /*****************************************************************************
     * Load tree data structures from HDF5
     *****************************************************************************/
    int read_tree_selection
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const CELL_IDX_T& pop_start,
     std::vector<neurotree_t> &tree_list,
     const std::vector<CELL_IDX_T>&  selection
     )
    {
      data::NamedAttrMap attr_values;
      
      read_cell_attribute_selection (comm, file_name, hdf5::TREES,
                                     pop_name, pop_start, 
                                     selection, attr_values);

      append_tree_list (pop_start, attr_values, tree_list);
    
      return 0;
    }

    
  }
}
