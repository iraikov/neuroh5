// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_tree.cc
///
///  Append tree structures to NeuroH5 file.
///
///  Copyright (C) 2016-2023 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>
#include <unistd.h>

#include <deque>
#include <vector>
#include <forward_list>

#include "neuroh5_types.hh"
#include "append_tree.hh"
#include "file_access.hh"
#include "rank_range.hh"
#include "range_sample.hh"
#include "dataset_num_elements.hh"
#include "exists_dataset.hh"
#include "enum_type.hh"
#include "path_names.hh"
#include "serialize_tree.hh"
#include "cell_index.hh"
#include "cell_attributes.hh"
#include "create_file_toplevel.hh"
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
     )
    {
      herr_t status; 

      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      size_t all_attr_size=0, all_sec_size=0,  all_topo_size=0;
      std::vector<size_t> attr_size_vector, sec_size_vector, topo_size_vector;

      attr_ptr.push_back(0);
      sec_ptr.push_back(0);
      topo_ptr.push_back(0);

      hsize_t local_ptr_size = std::distance(tree_list.begin(), tree_list.end());
      if (rank == size-1)
        {
          local_ptr_size=local_ptr_size+1;
        }

      std::vector<size_t> ptr_size_vector;
      ptr_size_vector.resize(size);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_SIZE_T, &ptr_size_vector[0], 1, MPI_SIZE_T, comm);
      throw_assert_nomsg(status == MPI_SUCCESS);

      for_each(tree_list.cbegin(),
               tree_list.cend(),
               [&] (const neurotree_t& tree)
               {

                 hsize_t attr_size=0, sec_size=0, topo_size=0;
                 
                 const CELL_IDX_T &idx = get<0>(tree);
                 const std::deque<SECTION_IDX_T> & src_vector=get<1>(tree);
                 const std::deque<SECTION_IDX_T> & dst_vector=get<2>(tree);
                 const std::deque<SECTION_IDX_T> & sections=get<3>(tree);
                 const std::deque<COORD_T> & xcoords=get<4>(tree);
                 const std::deque<COORD_T> & ycoords=get<5>(tree);
                 const std::deque<COORD_T> & zcoords=get<6>(tree);
                 const std::deque<REALVAL_T> & radiuses=get<7>(tree);
                 const std::deque<LAYER_IDX_T> & layers=get<8>(tree);
                 const std::deque<PARENT_NODE_IDX_T> & parents=get<9>(tree);
                 const std::deque<SWC_TYPE_T> & swc_types=get<10>(tree);
                 
                 topo_size = src_vector.size();
                 throw_assert_nomsg(src_vector.size() == topo_size);
                 throw_assert_nomsg(dst_vector.size() == topo_size);
                 
                 topo_ptr.push_back(topo_size+topo_ptr.back());
        
                 attr_size = xcoords.size();
                 throw_assert_nomsg(xcoords.size()  == attr_size);
                 throw_assert_nomsg(ycoords.size()  == attr_size);
                 throw_assert_nomsg(zcoords.size()  == attr_size);
                 throw_assert_nomsg(radiuses.size() == attr_size);
                 throw_assert_nomsg(layers.size()   == attr_size);
                 throw_assert_nomsg(parents.size()  == attr_size);
                 throw_assert_nomsg(swc_types.size()  == attr_size);
                 
                 attr_ptr.push_back(attr_size+attr_ptr.back());
                 
                 sec_size = sections.size();
                 sec_ptr.push_back(sec_size+sec_ptr.back());

                 all_index_vector.push_back(idx);
                 all_src_vector.insert(all_src_vector.end(),src_vector.begin(),src_vector.end());
                 all_dst_vector.insert(all_dst_vector.end(),dst_vector.begin(),dst_vector.end());
                 all_sections.insert(all_sections.end(),sections.begin(),sections.end());
                 all_xcoords.insert(all_xcoords.end(),xcoords.begin(),xcoords.end());
                 all_ycoords.insert(all_ycoords.end(),ycoords.begin(),ycoords.end());
                 all_zcoords.insert(all_zcoords.end(),zcoords.begin(),zcoords.end());
                 all_radiuses.insert(all_radiuses.end(),radiuses.begin(),radiuses.end());
                 all_layers.insert(all_layers.end(),layers.begin(),layers.end());
                 all_parents.insert(all_parents.end(),parents.begin(),parents.end());
                 all_swc_types.insert(all_swc_types.end(),swc_types.begin(),swc_types.end());
                 
                 all_attr_size = all_attr_size + attr_size;
                 all_sec_size  = all_sec_size + sec_size;
                 all_topo_size = all_topo_size + topo_size;
                 
               });

      return 0;
    }

    
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
     )
    {
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      size_t tree_list_size  = std::distance(tree_list.begin(), tree_list.end());
      throw_assert_nomsg(tree_list_size == 1); // singleton tree set
           
      for_each (tree_list.cbegin(), tree_list.cend(),
                [&] (const neurotree_t& tree)
                {

                  hsize_t attr_size=0, topo_size=0;
                  
                  const std::deque<SECTION_IDX_T> & src_vector=get<1>(tree);
                  const std::deque<SECTION_IDX_T> & dst_vector=get<2>(tree);
                  const std::deque<SECTION_IDX_T> & sections=get<3>(tree);
                  const std::deque<COORD_T> & xcoords=get<4>(tree);
                  const std::deque<COORD_T> & ycoords=get<5>(tree);
                  const std::deque<COORD_T> & zcoords=get<6>(tree);
                  const std::deque<REALVAL_T> & radiuses=get<7>(tree);
                  const std::deque<LAYER_IDX_T> & layers=get<8>(tree);
                  const std::deque<PARENT_NODE_IDX_T> & parents=get<9>(tree);
                  const std::deque<SWC_TYPE_T> & swc_types=get<10>(tree);
                  
                  topo_size = src_vector.size();
                  throw_assert_nomsg(src_vector.size() == topo_size);
                  throw_assert_nomsg(dst_vector.size() == topo_size);
                  
                  attr_size = xcoords.size();
                  throw_assert_nomsg(xcoords.size()  == attr_size);
                  throw_assert_nomsg(ycoords.size()  == attr_size);
                  throw_assert_nomsg(zcoords.size()  == attr_size);
                  throw_assert_nomsg(radiuses.size() == attr_size);
                  throw_assert_nomsg(layers.size()   == attr_size);
                  throw_assert_nomsg(parents.size()  == attr_size);
                  throw_assert_nomsg(swc_types.size()  == attr_size);
                  
                  all_src_vector.insert(all_src_vector.end(),src_vector.begin(),src_vector.end());
                  all_dst_vector.insert(all_dst_vector.end(),dst_vector.begin(),dst_vector.end());
                  all_sections.insert(all_sections.end(),sections.begin(),sections.end());
                  all_xcoords.insert(all_xcoords.end(),xcoords.begin(),xcoords.end());
                  all_ycoords.insert(all_ycoords.end(),ycoords.begin(),ycoords.end());
                  all_zcoords.insert(all_zcoords.end(),zcoords.begin(),zcoords.end());
                  all_radiuses.insert(all_radiuses.end(),radiuses.begin(),radiuses.end());
                  all_layers.insert(all_layers.end(),layers.begin(),layers.end());
                  all_parents.insert(all_parents.end(),parents.begin(),parents.end());
                  all_swc_types.insert(all_swc_types.end(),swc_types.begin(),swc_types.end());
                  
                });
      
      return 0;
    }


    herr_t size_tree_attributes
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& pop_name,
     const std::string& attr_name_space,
     hsize_t& attr_size,
     hsize_t& sec_size,
     hsize_t& topo_size
     )
    {
      hsize_t value_size, index_size;
      
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      if (rank == 0)
        {
          hid_t file = hdf5::open_file(comm, file_name);
          throw_assert_nomsg(file >= 0);

          string path;

          if (hdf5::exists_dataset(file, hdf5::cell_attribute_prefix(hdf5::TREES, pop_name)) > 0)
            {
              path = hdf5::cell_attribute_path (hdf5::TREES, pop_name, hdf5::X_COORD);
              hdf5::size_cell_attributes(comm, file, path, CellPtr (PtrOwner, hdf5::ATTR_PTR),
                                         attr_size, index_size, value_size);
              
              path = hdf5::cell_attribute_path (hdf5::TREES, pop_name, hdf5::SECTION);
              hdf5::size_cell_attributes(comm, file, path, CellPtr (PtrOwner, hdf5::SEC_PTR),
                                         sec_size, index_size, value_size);
              
              path = hdf5::cell_attribute_path (hdf5::TREES, pop_name, hdf5::SRCSEC);
              hdf5::size_cell_attributes(comm, file, path, CellPtr (PtrOwner, hdf5::SEC_PTR),
                                         topo_size, index_size, value_size);
              
            }
          else
            {
              attr_size = 0;
              sec_size = 0;
              topo_size = 0;
            }
          throw_assert_nomsg(hdf5::close_file(file) >= 0);
        }

      throw_assert_nomsg(MPI_Bcast(&attr_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Bcast(&sec_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Bcast(&topo_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);

      return 0;
    }
    
        
    /*****************************************************************************
     * Save tree data structures to HDF5
     *****************************************************************************/
    int append_trees
    (
     MPI_Comm                       comm,
     MPI_Comm                       io_comm,
     hid_t                          loc,
     const std::string&             pop_name,
     const CELL_IDX_T&              pop_start,
     std::forward_list<neurotree_t> &tree_list,
     const set<size_t>              &io_rank_set,
     CellPtr                        ptr_type,
     const size_t                   chunk_size,
     const size_t                   value_chunk_size
     )
    {
      herr_t status=0; 
      size_t io_size=0;

      size_t rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);
      
      size_t io_comm_size;
      throw_assert_nomsg(MPI_Comm_size(io_comm, (int*)&io_comm_size) == MPI_SUCCESS);
      
      
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;
      throw_assert(io_rank_set.size() > 0, "invalid I/O rank set");
      io_size = io_rank_set.size();
      throw_assert(io_comm_size == io_size, "mismatch between io_size and io_comm size");

      vector< pair<hsize_t,hsize_t> > ranges;
      mpi::rank_ranges(size, io_size, ranges);
      
      // Determine I/O ranks to which to send the values
      vector <size_t> io_dests(size); 
      for (size_t r=0; r<size; r++)
        {
          for (size_t i=ranges.size()-1; i>=0; i--)
            {
              if (ranges[i].first <= r)
                {
                  io_dests[r] = *std::next(io_rank_set.begin(), i);
                  break;
                }
            }
        }
      
      std::forward_list<neurotree_t> local_tree_list;
      {
        std::vector<char> sendbuf; 
        std::vector<int> sendcounts, sdispls;
        vector<int> recvcounts, rdispls;
        vector<char> recvbuf;
        
        rank_t dst_rank = io_dests[rank];
        map <rank_t, map<CELL_IDX_T, neurotree_t> > rank_tree_map;
        for_each(tree_list.cbegin(),
                 tree_list.cend(),
                 [&] (const neurotree_t& tree)
                 {
                   set <rank_t> dst_rank_set;
                   
                   const CELL_IDX_T &idx = get<0>(tree);
                   CELL_IDX_T gid = idx;
                   
                   map<CELL_IDX_T, neurotree_t> &tree_map = rank_tree_map[dst_rank];
                   tree_map.insert(make_pair(gid, tree));
                 });
          
        data::serialize_rank_tree_map (size, rank, rank_tree_map, sendcounts, sendbuf, sdispls);
        
        throw_assert_nomsg(mpi::alltoallv_vector<char>(comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                       recvcounts, rdispls, recvbuf) >= 0);
        sendbuf.clear();
        sendbuf.shrink_to_fit();
        
        data::deserialize_rank_tree_list (size, recvbuf, recvcounts, rdispls,
                                          local_tree_list);
      }
      
      
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
                                                 local_tree_list,
                                                 all_src_vector, all_dst_vector,
                                                 all_xcoords, all_ycoords, all_zcoords, 
                                                 all_radiuses, all_layers, all_sections,
                                                 all_parents, all_swc_types);
        }
      else
        {
          status = build_tree_datasets(comm,
                                       local_tree_list,
                                       sec_ptr, topo_ptr, attr_ptr,
                                       all_index_vector, all_src_vector, all_dst_vector,
                                       all_xcoords, all_ycoords, all_zcoords, 
                                       all_radiuses, all_layers, all_sections,
                                       all_parents, all_swc_types);
          throw_assert_nomsg(status >= 0);
        }

      local_tree_list.clear();
      

      const data::optional_hid dflt_data_type;
      const data::optional_hid coord_data_type(COORD_H5_NATIVE_T);
      const data::optional_hid layer_data_type(LAYER_IDX_H5_NATIVE_T);
      const data::optional_hid parent_node_data_type(PARENT_NODE_IDX_H5_NATIVE_T);
      const data::optional_hid section_data_type(SECTION_IDX_H5_NATIVE_T);
      const data::optional_hid swc_data_type(SWC_TYPE_H5_NATIVE_T);

      string attr_ptr_owner_path = hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::X_COORD) + "/" + hdf5::ATTR_PTR;
      string sec_ptr_owner_path  = hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SRCSEC) + "/" + hdf5::SEC_PTR;

      
      hid_t file;

      if (is_io_rank)
        {

          hid_t file = H5Iget_file_id(loc);
          throw_assert(file >= 0,
                       "append_trees: invalid file handle");
          
          append_cell_index (io_comm, file, pop_name, pop_start,
                             hdf5::TREES, all_index_vector);
          
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::X_COORD,
                                 all_index_vector, attr_ptr, all_xcoords,
                                 coord_data_type, IndexShared,
                                 CellPtr (PtrOwner, hdf5::ATTR_PTR),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::Y_COORD,
                                 all_index_vector, attr_ptr, all_ycoords,
                                 coord_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::Z_COORD,
                                 all_index_vector, attr_ptr, all_zcoords,
                                 coord_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::RADIUS,
                                 all_index_vector, attr_ptr, all_radiuses,
                                 dflt_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::LAYER,
                                 all_index_vector, attr_ptr, all_layers,
                                 layer_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::PARENT,
                                 all_index_vector, attr_ptr, all_parents,
                                 parent_node_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::SWCTYPE,
                                 all_index_vector, attr_ptr, all_swc_types,
                                 swc_data_type, IndexShared,
                                 CellPtr (PtrShared, attr_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::SRCSEC,
                                 all_index_vector, topo_ptr, all_src_vector,
                                 section_data_type, IndexShared,
                                 CellPtr (PtrOwner, hdf5::SEC_PTR),
                                 chunk_size, value_chunk_size);
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::DSTSEC,
                                 all_index_vector, topo_ptr, all_dst_vector,
                                 section_data_type, IndexShared,
                                 CellPtr (PtrShared, sec_ptr_owner_path),
                                 chunk_size, value_chunk_size);
          
          append_cell_attribute (file, hdf5::TREES, pop_name, pop_start, hdf5::SECTION,
                                 all_index_vector, sec_ptr, all_sections,
                                 section_data_type, IndexShared,
                                 CellPtr (PtrOwner, hdf5::SEC_PTR),
                                 chunk_size, value_chunk_size);


          status = H5Fclose(file);
          throw_assert(status == 0, "append_trees: unable to close HDF5 file");

        }


      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
      
      return 0;
    }

    int append_trees
    (
     MPI_Comm                       comm,
     const std::string&             file_name,
     const std::string&             pop_name,
     const CELL_IDX_T&              pop_start,
     std::forward_list<neurotree_t> &tree_list,
     size_t                         io_size,
     const size_t                   chunk_size,
     const size_t                   value_chunk_size
     )
    {
      herr_t status;

      size_t rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);
      
      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);

      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;
      
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1, color;
      
      // Am I an I/O rank?
      if (is_io_rank)
        {
          color = io_color;
        }
      else
        {
          color = 0;
        }
      MPI_Comm_split(comm,color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

      if (access( file_name.c_str(), F_OK ) != 0)
        {
          vector <string> groups;
          groups.push_back (hdf5::POPULATIONS);
          status = hdf5::create_file_toplevel (io_comm, file_name, groups);
        }
        else
          {
            status = 0;
          }
      throw_assert(status == 0,
                   "append_trees: unable to create toplevel groups in file");
      
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");

      hid_t file = hdf5::open_file(io_comm, file_name, true, true);
      
      status = append_trees(comm, io_comm, file, pop_name, pop_start, tree_list,
                            io_rank_set,  CellPtr(PtrOwner), chunk_size, value_chunk_size);
      throw_assert_nomsg(status >= 0);

      hdf5::close_file(file);
      
      throw_assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS,
                   "append_trees: error in MPI_Comm_free");
      
      return 0;
    }

    
  }
}
