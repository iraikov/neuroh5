// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_tree.cc
///
///  Read tree structures.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>

#include <cassert>
#include <vector>

#include "neuroh5_types.hh"
#include "file_access.hh"
#include "rank_range.hh"
#include "dataset_num_elements.hh"
#include "enum_type.hh"
#include "path_names.hh"
#include "read_template.hh"

namespace neuroh5
{

  namespace cell
  {

    void append_tree_list
    (
     const size_t start,
     const size_t num_trees,
     const CELL_IDX_T pop_start,
     std::vector<SEC_PTR_T>& sec_ptr,
     std::vector<TOPO_PTR_T>& topo_ptr,
     std::vector<ATTR_PTR_T>& attr_ptr,
     std::vector<CELL_IDX_T>& all_gid_vector,
     std::vector<SECTION_IDX_T>& all_src_vector,
     std::vector<SECTION_IDX_T>& all_dst_vector,
     std::vector<SECTION_IDX_T>& all_sections,
     std::vector<COORD_T>& all_xcoords,
     std::vector<COORD_T>& all_ycoords,
     std::vector<COORD_T>& all_zcoords,
     std::vector<REALVAL_T>& all_radiuses,
     std::vector<LAYER_IDX_T>& all_layers,
     std::vector<PARENT_NODE_IDX_T>& all_parents,
     std::vector<SWC_TYPE_T>& all_swc_types,
     std::vector<neurotree_t> &tree_list)
    {
      for (size_t i=0; i<num_trees; i++)
        {
          hsize_t topo_start = topo_ptr[i]-topo_ptr[0]; size_t topo_block = topo_ptr[i+1]-topo_ptr[0]-topo_start;

          vector<SECTION_IDX_T>::const_iterator src_first = all_src_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator src_last  = all_src_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_src_vector;
          tree_src_vector.insert(tree_src_vector.begin(),src_first,src_last);
        
          vector<SECTION_IDX_T>::const_iterator dst_first = all_dst_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator dst_last  = all_dst_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_dst_vector;
          tree_dst_vector.insert(tree_dst_vector.begin(),dst_first,dst_last);
        
          hsize_t sec_start = sec_ptr[i]-sec_ptr[0]; size_t sec_block = sec_ptr[i+1]-sec_ptr[0]-sec_start;
        
          vector<SECTION_IDX_T>::const_iterator sec_first = all_sections.begin() + sec_start;
          vector<SECTION_IDX_T>::const_iterator sec_last  = all_sections.begin() + sec_start + sec_block;
          vector<SECTION_IDX_T> tree_sections;
          tree_sections.insert(tree_sections.begin(),sec_first,sec_last);
        
          hsize_t attr_start = attr_ptr[i]-attr_ptr[0]; size_t attr_block = attr_ptr[i+1]-attr_ptr[0]-attr_start;

          vector<COORD_T>::const_iterator xcoords_first = all_xcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator xcoords_last  = all_xcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_xcoords;
          tree_xcoords.insert(tree_xcoords.begin(),xcoords_first,xcoords_last);

          vector<COORD_T>::const_iterator ycoords_first = all_ycoords.begin() + attr_start;
          vector<COORD_T>::const_iterator ycoords_last  = all_ycoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_ycoords;
          tree_ycoords.insert(tree_ycoords.begin(),ycoords_first,ycoords_last);

          vector<COORD_T>::const_iterator zcoords_first = all_zcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator zcoords_last  = all_zcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_zcoords;
          tree_zcoords.insert(tree_zcoords.begin(),zcoords_first,zcoords_last);

          vector<REALVAL_T>::const_iterator radiuses_first = all_radiuses.begin() + attr_start;
          vector<REALVAL_T>::const_iterator radiuses_last  = all_radiuses.begin() + attr_start + attr_block;
          vector<REALVAL_T> tree_radiuses;
          tree_radiuses.insert(tree_radiuses.begin(),radiuses_first,radiuses_last);

          vector<LAYER_IDX_T>::const_iterator layers_first = all_layers.begin() + attr_start;
          vector<LAYER_IDX_T>::const_iterator layers_last  = all_layers.begin() + attr_start + attr_block;
          vector<LAYER_IDX_T> tree_layers;
          tree_layers.insert(tree_layers.begin(),layers_first,layers_last);

          vector<PARENT_NODE_IDX_T>::const_iterator parents_first = all_parents.begin() + attr_start;
          vector<PARENT_NODE_IDX_T>::const_iterator parents_last  = all_parents.begin() + attr_start + attr_block;
          vector<PARENT_NODE_IDX_T> tree_parents;
          tree_parents.insert(tree_parents.begin(),parents_first,parents_last);
        
          vector<SWC_TYPE_T>::const_iterator swc_types_first = all_swc_types.begin() + attr_start;
          vector<SWC_TYPE_T>::const_iterator swc_types_last  = all_swc_types.begin() + attr_start + attr_block;
          vector<SWC_TYPE_T> tree_swc_types;
          tree_swc_types.insert(tree_swc_types.begin(),swc_types_first,swc_types_last);

          CELL_IDX_T gid = pop_start+all_gid_vector[i];
          tree_list.push_back(make_tuple(gid,tree_src_vector,tree_dst_vector,tree_sections,
                                         tree_xcoords,tree_ycoords,tree_zcoords,
                                         tree_radiuses,tree_layers,tree_parents,
                                         tree_swc_types));
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
     const CELL_IDX_T pop_start,
     std::vector<neurotree_t> &tree_list,
     size_t offset = 0,
     size_t numitems = 0,
     bool collective = true
     )
    {
      herr_t status; hid_t rapl;

      std::vector<SEC_PTR_T> sec_ptr;
      std::vector<TOPO_PTR_T> topo_ptr;
      std::vector<ATTR_PTR_T> attr_ptr;

      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      /* Create HDF5 enumerated type for reading SWC type information */
      hid_t hdf5_swc_type = hdf5::create_H5Tenum<SWC_TYPE_T> (swc_type_enumeration);

      /* Create property list for collective dataset operations. */
      rapl = H5Pcreate (H5P_DATASET_XFER);
      if (collective)
        {
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
        }

      // TODO; create separate functions for opening HDF5 file for reading and writing
      hid_t file = hdf5::open_file(comm, file_name);
      size_t dset_size = hdf5::dataset_num_elements(comm, file, hdf5::cell_attribute_path(hdf5::TREES, string(pop_name), hdf5::ATTR_PTR))-1;
      size_t read_size = 0;
      if (numitems > 0) 
        {
          if (offset < dset_size)
            {
              read_size = min(numitems*size, dset_size-offset);
            }
          else
            {
              read_size = 0;
            }
        }
      else
        {
          read_size = dset_size;
        }

      if (read_size > 0)
        {
          // determine which blocks of block_ptr are read by which rank
          vector< pair<hsize_t,hsize_t> > ranges;
          mpi::rank_ranges(read_size, size, ranges);
        
          hsize_t start = ranges[rank].first + offset;
          hsize_t end   = start + ranges[rank].second;
          hsize_t block = end - start + 1;

          if (block > 0)
            {
              // allocate buffer and memory dataspace
              attr_ptr.resize(block);
            
              status = hdf5::read<ATTR_PTR_T> (file,
                                               hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::ATTR_PTR),
                                               start, block,
                                               ATTR_PTR_H5_NATIVE_T,
                                               attr_ptr, rapl);
              assert(status >= 0);
            
              sec_ptr.resize(block);
              status = hdf5::read<SEC_PTR_T> (file,
                                              hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SEC_PTR),
                                              start, block,
                                              SEC_PTR_H5_NATIVE_T,
                                              sec_ptr, rapl);
              assert(status >= 0);
            
              topo_ptr.resize(block);
              status = hdf5::read<TOPO_PTR_T> (file,
                                               hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::TOPO_PTR),
                                               start, block,
                                               TOPO_PTR_H5_NATIVE_T,
                                               topo_ptr, rapl);
              assert(status >= 0);

              std::vector<CELL_IDX_T> gid_vector;
              std::vector<SECTION_IDX_T> src_vector, dst_vector;
              std::vector<SECTION_IDX_T> sections;
              std::vector<COORD_T> xcoords;
              std::vector<COORD_T> ycoords;
              std::vector<COORD_T> zcoords;
              std::vector<REALVAL_T> radiuses;
              std::vector<LAYER_IDX_T> layers;
              std::vector<PARENT_NODE_IDX_T> parents;
              std::vector<SWC_TYPE_T> swc_types;
            
              hsize_t topo_start = topo_ptr[0];
              size_t topo_block = topo_ptr.back()-topo_start;
            
              gid_vector.resize(block-1);
              status = hdf5::read<CELL_IDX_T> (file,
                                               hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::CELL_INDEX),
                                               start, block-1,
                                               CELL_IDX_H5_NATIVE_T,
                                               gid_vector, rapl);
            
              src_vector.resize(topo_block);
              status = hdf5::read<SECTION_IDX_T> (file,
                                                 hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SRCSEC),
                                                 topo_start, topo_block,
                                                 SECTION_IDX_H5_NATIVE_T,
                                                 src_vector, rapl);
              assert(status == 0);
              dst_vector.resize(topo_block);
              status = hdf5::read<SECTION_IDX_T> (file,
                                                 hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::DSTSEC),
                                                 topo_start, topo_block,
                                                 SECTION_IDX_H5_NATIVE_T,
                                                 dst_vector, rapl);
              assert(status == 0);
            
              hsize_t sec_start = sec_ptr[0];
              size_t sec_block = sec_ptr.back()-sec_start;
            
              sections.resize(sec_block);
              status = hdf5::read<SECTION_IDX_T> (file,
                                                 hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SECTION),
                                                 sec_start, sec_block,
                                                 SECTION_IDX_H5_NATIVE_T,
                                                 sections, rapl);
              assert(status == 0);
            
            
            
              hsize_t attr_start = attr_ptr[0];
              size_t attr_block = attr_ptr.back()-attr_start;
            
              xcoords.resize(attr_block);
              status = hdf5::read<COORD_T> (file,
                                           hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::X_COORD),
                                           attr_start, attr_block,
                                           COORD_H5_NATIVE_T,
                                           xcoords, rapl);
              assert(status == 0);
              ycoords.resize(attr_block);
              status = hdf5::read<COORD_T> (file,
                                           hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::Y_COORD),
                                           attr_start, attr_block,
                                           COORD_H5_NATIVE_T,
                                           ycoords, rapl);
              assert(status == 0);
              zcoords.resize(attr_block);
              status = hdf5::read<COORD_T> (file,
                                           hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::Z_COORD),
                                           attr_start, attr_block,
                                           COORD_H5_NATIVE_T,
                                           zcoords, rapl);
              assert(status == 0);
              radiuses.resize(attr_block);
              status = hdf5::read<REALVAL_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::RADIUS),
                                             attr_start, attr_block,
                                             REAL_H5_NATIVE_T,
                                             radiuses, rapl);
              assert(status == 0);
              layers.resize(attr_block);
              status = hdf5::read<LAYER_IDX_T> (file,
                                               hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::LAYER),
                                               attr_start, attr_block,
                                               LAYER_IDX_H5_NATIVE_T,
                                               layers, rapl);
              assert(status == 0);
              parents.resize(attr_block);
              status = hdf5::read<PARENT_NODE_IDX_T> (file,
                                                     hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::PARENT),
                                                     attr_start, attr_block,
                                                     PARENT_NODE_IDX_H5_NATIVE_T,
                                                     parents, rapl);
              assert(status == 0);
              swc_types.resize(attr_block);
              status = hdf5::read<SWC_TYPE_T> (file,
                                               hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SWCTYPE),
                                              attr_start, attr_block,
                                              hdf5_swc_type,
                                              swc_types, rapl);
              assert(status == 0);

              append_tree_list(start, block-1, pop_start,
                               sec_ptr, topo_ptr, attr_ptr,
                               gid_vector, src_vector, dst_vector, sections,
                               xcoords, ycoords, zcoords,
                               radiuses, layers, parents,
                               swc_types, tree_list);
            }
        }
    
      status = hdf5::close_file (file);
      assert(status == 0);
      status = H5Pclose(rapl);
      assert(status == 0);
      status = H5Tclose(hdf5_swc_type);
      assert(status == 0);
    
      return 0;
    }
  }
}
