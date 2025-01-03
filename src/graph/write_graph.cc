// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file write_graph.cc
///
///  Top-level functions for writing graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2024 Project NeuroH5.
//==============================================================================



#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_populations.hh"
#include "write_graph.hh"
#include "write_projection.hh"
#include "path_names.hh"
#include "sort_permutation.hh"
#include "serialize_edge.hh"
#include "throw_assert.hh"
#include "debug.hh"

#include <vector>
#include <map>

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {

    // Assign each node to a rank 
    static void compute_node_rank_map
    (
     size_t num_ranks,
     size_t num_nodes,
     map< NODE_IDX_T, rank_t > &node_rank_map
     )
    {
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < num_nodes; i++)
        {
            node_rank_map.insert(make_pair(i, i%num_ranks));
        }
    }
    
    int write_graph
    (
     MPI_Comm         all_comm,
     const int        io_size_arg,
     const string&    file_name,
     const string&    src_pop_name,
     const string&    dst_pop_name,
     const std::map <std::string, std::pair <size_t, data::AttrIndex > >& edge_attr_index,
     const edge_map_t&  input_edge_map,
     const hsize_t    chunk_size
     )
    {
      size_t io_size;
      size_t num_edges = 0;
      
      // read the population info
      set< pair<pop_t, pop_t> > pop_pairs;
      pop_range_map_t pop_ranges;
      pop_label_map_t pop_labels;
      size_t src_pop_idx, dst_pop_idx; bool src_pop_set=false, dst_pop_set=false;
      size_t total_num_nodes;
      size_t dst_start, dst_end;
      size_t src_start, src_end;

      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };

      int size, rank;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);

      if (size < io_size_arg)
        {
          io_size = size > 0 ? size : 1;
        }
      else
        {
          io_size = io_size_arg > 0 ? io_size_arg : 1;
        }
      //FIXME: throw_assert_nomsg(io::hdf5::read_population_combos(comm, file_name, pop_pairs) >= 0);
      throw_assert_nomsg(cell::read_population_ranges(all_comm, file_name, pop_ranges, total_num_nodes) >= 0);
      throw_assert_nomsg(cell::read_population_labels(all_comm, file_name, pop_labels) >= 0);
      
      for (auto& x: pop_labels)
        {
          if (src_pop_name == get<1>(x))
            {
              src_pop_idx = get<0>(x);
              src_pop_set = true;
            }
          if (dst_pop_name == get<1>(x))
            {
              dst_pop_idx = get<0>(x);
              dst_pop_set = true;
            }
        }
      throw_assert_nomsg(dst_pop_set && src_pop_set);
      
      dst_start = pop_ranges[dst_pop_idx].start;
      dst_end   = dst_start + pop_ranges[dst_pop_idx].count;
      src_start = pop_ranges[src_pop_idx].start;
      src_end   = src_start + pop_ranges[src_pop_idx].count;

      total_num_nodes = 0;
      vector< NODE_IDX_T > node_index;

      { // Determine the destination node indices present in the input
        // edge map across all ranks
        size_t num_nodes = input_edge_map.size();
        vector<size_t> sendbuf_num_nodes(size, num_nodes);
        vector<size_t> recvbuf_num_nodes(size);
        vector<int> recvcounts(size, 0);
        vector<int> displs(size+1, 0);
        throw_assert_nomsg(MPI_Allgather(&sendbuf_num_nodes[0], 1, MPI_SIZE_T,
                                         &recvbuf_num_nodes[0], 1, MPI_SIZE_T, all_comm)
                           == MPI_SUCCESS);
        throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
        for (size_t p=0; p<size; p++)
          {
            total_num_nodes = total_num_nodes + recvbuf_num_nodes[p];
            displs[p+1] = displs[p] + recvbuf_num_nodes[p];
            recvcounts[p] = recvbuf_num_nodes[p];
          }
        
        vector< NODE_IDX_T > local_node_index;
        for (auto iter: input_edge_map)
          {
            NODE_IDX_T dst          = iter.first;
            local_node_index.push_back(dst);
          }
        
        node_index.resize(total_num_nodes,0);
        throw_assert_nomsg(MPI_Allgatherv(&local_node_index[0], num_nodes, MPI_NODE_IDX_T,
                                          &node_index[0], &recvcounts[0], &displs[0], MPI_NODE_IDX_T,
                                          all_comm) == MPI_SUCCESS);
        throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);

        vector<size_t> p = sort_permutation(node_index, compare_nodes);
        apply_permutation_in_place(node_index, p);
      }

      throw_assert_nomsg(node_index.size() == total_num_nodes);

      if (total_num_nodes == 0)
        {
          throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
          return 0;
        }

      
      // Create an I/O communicator
      MPI_Comm  io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      if ((rank_t)rank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      
      // A vector that maps nodes to compute ranks
      map< NODE_IDX_T, rank_t > node_rank_map;
      compute_node_rank_map(io_size, total_num_nodes, node_rank_map);

      // construct a map where each set of edges are arranged by destination I/O rank
      rank_edge_map_t rank_edge_map;
      for (auto iter : input_edge_map)
        {
          NODE_IDX_T dst = iter.first;
          // all source/destination node IDs must be in range
          throw_assert_nomsg(dst_start <= dst && dst <= dst_end);
          edge_tuple_t& et        = iter.second;
          vector<NODE_IDX_T>& v   = get<0>(et);
          vector <AttrVal> & va   = get<1>(et);

          vector<NODE_IDX_T> adj_vector;
          for (auto & src: v)
            {
              if (!(src_start <= src && src <= src_end))
                {
                  printf("src = %u src_start = %lu src_end = %lu\n", src, src_start, src_end);
                }
              throw_assert_nomsg(src_start <= src && src <= src_end);
              adj_vector.push_back(src - src_start);
              num_edges++;
            }
          vector<size_t> p = sort_permutation(adj_vector, compare_nodes);

          apply_permutation_in_place(adj_vector, p);

          for (auto & a : va)
            {
              
              for (size_t i=0; i<a.size_attr_vec<float>(); i++)
                {
                  throw_assert_nomsg(a.float_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.float_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<uint8_t>(); i++)
                {
                  throw_assert_nomsg(a.uint8_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.uint8_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<uint16_t>(); i++)
                {
                  throw_assert_nomsg(a.uint16_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.uint16_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<uint32_t>(); i++)
                {
                  throw_assert_nomsg(a.uint32_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.uint32_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<int8_t>(); i++)
                {
                  throw_assert_nomsg(a.int8_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.int8_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<int16_t>(); i++)
                {
                  throw_assert_nomsg(a.int16_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.int16_values[i], p);
                }
              for (size_t i=0; i<a.size_attr_vec<int32_t>(); i++)
                {
                  throw_assert_nomsg(a.int32_values[i].size() == adj_vector.size());
                  apply_permutation_in_place(a.int32_values[i], p);
                }
            }
          
          auto it = node_rank_map.find(dst);
          throw_assert_nomsg(it != node_rank_map.end());
          size_t dst_rank = it->second;
          edge_tuple_t& et1 = rank_edge_map[dst_rank][dst];
          vector<NODE_IDX_T> &src_vec = get<0>(et1);
          src_vec.insert(src_vec.end(),adj_vector.begin(),adj_vector.end());
          vector <AttrVal> &edge_attr_vec = get<1>(et1);
          edge_attr_vec.resize(va.size());
          
          size_t i=0;
          for (auto & edge_attr : edge_attr_vec)
            {
              AttrVal& a = va[i];

              edge_attr.float_values.resize(a.size_attr_vec<float>());
              edge_attr.uint8_values.resize(a.size_attr_vec<uint8_t>());
              edge_attr.uint16_values.resize(a.size_attr_vec<uint16_t>());
              edge_attr.uint32_values.resize(a.size_attr_vec<uint32_t>());
              edge_attr.int8_values.resize(a.size_attr_vec<int8_t>());
              edge_attr.int16_values.resize(a.size_attr_vec<int16_t>());
              edge_attr.int32_values.resize(a.size_attr_vec<int32_t>());
              edge_attr.append(a);
              i++;
            }
        }


      // send buffer and structures for MPI Alltoall operation
      vector<char> sendbuf;
      vector<int> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);

      // Create serialized object with the edges of vertices for the respective I/O rank
      size_t num_packed_edges = 0; 

      data::serialize_rank_edge_map (size, rank, rank_edge_map, num_packed_edges,
                                     sendcounts, sendbuf, sdispls);
      rank_edge_map.clear();
      
      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks receive zero),
      //    and creates sendcounts and sdispls arrays
      
      throw_assert_nomsg(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) == MPI_SUCCESS);
      
      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      
      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < size; p++)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      vector<char> recvbuf;
      recvbuf.resize(recvbuf_size > 0 ? recvbuf_size : 1, 0);
      
      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      throw_assert_nomsg(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_CHAR,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_CHAR,
                           all_comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
      sendbuf.clear();
      sendcounts.clear();
      sdispls.clear();

      size_t num_unpacked_edges = 0, num_unpacked_nodes = 0;
      edge_map_t prj_edge_map;
      if (recvbuf_size > 0)
        {
          data::deserialize_rank_edge_map (size, recvbuf, recvcounts, rdispls, 
                                           prj_edge_map, num_unpacked_nodes,
                                           num_unpacked_edges);
        }

      if ((rank_t)rank < io_size)
        {
          hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
          throw_assert_nomsg(fapl >= 0);
#ifdef HDF5_IS_PARALLEL
          throw_assert_nomsg(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
#endif
          
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
          throw_assert_nomsg(file >= 0);
          
          write_projection (io_comm, file, src_pop_name, dst_pop_name,
                            src_start, src_end, dst_start, dst_end,
                            num_unpacked_edges, prj_edge_map, edge_attr_index,
                            chunk_size);
          
          throw_assert_nomsg(H5Fclose(file) >= 0);
          throw_assert_nomsg(H5Pclose(fapl) >= 0);
          throw_assert_nomsg(MPI_Barrier(io_comm) == MPI_SUCCESS);
        }

      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);
      MPI_Barrier(all_comm);

      return 0;
    }
  }
}
