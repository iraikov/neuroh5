// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_tree.cc
///
///  Read and scatter tree structures.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>

#include <vector>
#include <map>
#include <set>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"
#include "range_sample.hh"
#include "append_tree_map.hh"
#include "append_rank_tree_map.hh"
#include "append_rank_attr_map.hh"
#include "alltoallv_template.hh"
#include "serialize_tree.hh"
#include "serialize_cell_attributes.hh"
#include "dataset_num_elements.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "throw_assert.hh"
#include "debug.hh"

using namespace std;

namespace neuroh5
{

  namespace cell
  {
  
    /*****************************************************************************
     * Load tree data structures from HDF5 and scatter to all ranks
     *****************************************************************************/
    int scatter_read_trees
    (
     MPI_Comm                        comm,
     const string                   &file_name,
     const int                       io_size,
     const vector<string>           &attr_name_spaces,
     // A vector that maps nodes to compute ranks
     const node_rank_map_t           &node_rank_map,
     const string                    &pop_name,
     const CELL_IDX_T                 pop_start,
     map<CELL_IDX_T, neurotree_t>    &tree_map,
     map<string, data::NamedAttrMap> &attr_maps,
     size_t offset,
     size_t numitems
     )
    {
      std::vector<char> sendbuf; 
      std::vector<int> sendcounts, sdispls;
    
      MPI_Comm all_comm;
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      throw_assert_nomsg(io_size > 0);

      throw_assert(MPI_Comm_dup(comm, &(all_comm)) == MPI_SUCCESS,
                   "scatter_read_tree: unable to duplicate MPI communicator");
    
      int srank, ssize; size_t rank=0, size=0, io_rank;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &ssize) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &srank) == MPI_SUCCESS);
      throw_assert_nomsg(srank >= 0);
      throw_assert_nomsg(ssize > 0);
      rank = srank;
      size = ssize;

      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = (io_rank_set.find(rank) != io_rank_set.end());

      // Am I an I/O rank?
      if (is_io_rank)
        {
          throw_assert(MPI_Comm_split(all_comm,io_color,rank,&io_comm) == MPI_SUCCESS,
                       "scatter_read_trees: error in MPI_Comm_split");
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
          throw_assert_nomsg(MPI_Comm_rank(io_comm, &srank) == MPI_SUCCESS);
          io_rank = srank;
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
    
      sendcounts.resize(size,0);
      sdispls.resize(size,0);
      sendbuf.resize(0);

#ifdef NEUROH5_DEBUG
      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
#endif

      if (is_io_rank)
        {
          map <rank_t, map<CELL_IDX_T, neurotree_t> > rank_tree_map;
          {
            data::NamedAttrMap attr_values;
            set <string> attr_mask;
            
            read_cell_attributes (io_comm, file_name, hdf5::TREES, attr_mask,
                                  pop_name, pop_start, attr_values,
                                  offset, numitems * size);

            data::append_rank_tree_map(attr_values, node_rank_map, rank_tree_map);
          }
          data::serialize_rank_tree_map (size, rank, rank_tree_map, sendcounts, sendbuf, sdispls);
        }


#ifdef NEUROH5_DEBUG
      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
#endif

      throw_assert_nomsg(MPI_Barrier(io_comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      {
        vector<int> recvcounts, rdispls;
        vector<char> recvbuf;

        throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                       recvcounts, rdispls, recvbuf) >= 0);
        sendbuf.clear();
        sendbuf.shrink_to_fit();

        if (recvbuf.size() > 0)
          {
            data::deserialize_rank_tree_map (size, recvbuf, recvcounts, rdispls, tree_map);
          }
        recvbuf.clear();
        recvbuf.shrink_to_fit();
      }

      for (string attr_name_space : attr_name_spaces)
        {
          data::NamedAttrMap attr_map;
          set <string> attr_mask;

          scatter_read_cell_attributes(all_comm, file_name, io_size,
                                       attr_name_space, attr_mask, node_rank_map,
                                       pop_name, pop_start, attr_map,
                                       offset, numitems);
          attr_maps.insert(make_pair(attr_name_space, attr_map));
        }

      throw_assert_nomsg(MPI_Comm_free(&all_comm) == MPI_SUCCESS);
    
      return 0;
    }


    int scatter_read_tree_selection
    (
     MPI_Comm                        all_comm,
     const string                   &file_name,
     const int                       io_size,
     const vector<string>           &attr_name_spaces,
     // A vector that maps nodes to compute ranks
     const string                    &pop_name,
     const CELL_IDX_T                 pop_start,
     const std::vector<CELL_IDX_T>&  selection,
     map<CELL_IDX_T, neurotree_t>    &tree_map,
     map<string, data::NamedAttrMap> &attr_maps
     )
    {

      data::NamedAttrMap attr_values;
      set <string> attr_mask;

      scatter_read_cell_attribute_selection (all_comm, file_name, io_size, hdf5::TREES, attr_mask,
                                             pop_name, pop_start, selection, attr_values);
      append_tree_map(attr_values, tree_map);

      for (string attr_name_space : attr_name_spaces)
        {
          scatter_read_cell_attribute_selection(all_comm, file_name,  io_size,
                                                attr_name_space, attr_mask, 
                                                pop_name, pop_start, selection, 
                                                attr_maps[attr_name_space]);

        }
#ifdef NEUROH5_DEBUG
      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
#endif    
      return 0;
    }

  }
}
