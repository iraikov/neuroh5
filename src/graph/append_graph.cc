// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_graph.cc
///
///  Top-level functions for appending edge information to graphs in
///  DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_populations.hh"
#include "append_graph.hh"
#include "append_projection.hh"
#include "path_names.hh"
#include "sort_permutation.hh"
#include "serialize_edge.hh"

#include <vector>
#include <map>

#undef NDEBUG
#include <cassert>

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
      size_t num_partitions;
      hsize_t remainder=0, offset=0, buckets=0;
      if (num_ranks > 0)
        {
          num_partitions = num_ranks;
        }
      else
        {
          num_partitions = 1;
        }
      for (size_t i=0; i<num_partitions; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_partitions - i;
          for (size_t j = 0; j < remainder / buckets; j++)
            {
              node_rank_map.insert(make_pair(offset+j, i));
            }
          offset    += remainder / buckets;
        }
    }

    
    int append_graph
    (
     MPI_Comm         all_comm,
     const int        io_size_arg,
     const string&    file_name,
     const string&    src_pop_name,
     const string&    dst_pop_name,
     const map <string, vector <vector <string> > >& edge_attr_names,
     const edge_map_t&  input_edge_map
     )
    {
      size_t io_size;
      size_t num_edges = 0;
      
      // read the population info
      set< pair<pop_t, pop_t> > pop_pairs;
      vector<pop_range_t> pop_vector;
      vector<pair <pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      size_t src_pop_idx, dst_pop_idx; bool src_pop_set=false, dst_pop_set=false;
      size_t total_num_nodes;
      size_t dst_start, dst_end;
      size_t src_start, src_end;

      int size, rank;
      assert(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);

      if (size < io_size_arg)
        {
          io_size = size > 0 ? size : 1;
        }
      else
        {
          io_size = io_size_arg > 0 ? io_size_arg : 1;
        }
      DEBUG("Task ",rank,": ","append_graph: io_size = ",io_size,"\n");
      DEBUG("Task ",rank,": ","append_graph: prior to reading population ranges\n");
      //FIXME: assert(io::hdf5::read_population_combos(comm, file_name, pop_pairs) >= 0);
      assert(cell::read_population_ranges(all_comm, file_name,
                                          pop_ranges, pop_vector, total_num_nodes) >= 0);
      DEBUG("Task ",rank,": ","append_graph: prior to reading population labels\n");
      assert(cell::read_population_labels(all_comm, file_name, pop_labels) >= 0);
      DEBUG("Task ",rank,": ","append_graph: read population labels\n");
      
      for (size_t i=0; i< pop_labels.size(); i++)
        {
          if (src_pop_name == get<1>(pop_labels[i]))
            {
              src_pop_idx = get<0>(pop_labels[i]);
              src_pop_set = true;
            }
          if (dst_pop_name == get<1>(pop_labels[i]))
            {
              dst_pop_idx = get<0>(pop_labels[i]);
              dst_pop_set = true;
            }
        }
      assert(dst_pop_set && src_pop_set);
      
      dst_start = pop_vector[dst_pop_idx].start;
      dst_end   = dst_start + pop_vector[dst_pop_idx].count;
      src_start = pop_vector[src_pop_idx].start;
      src_end   = src_start + pop_vector[src_pop_idx].count;
      
      DEBUG("Task ",rank,": ","append_graph: dst_start = ", dst_start, " dst_end = ", dst_end, "\n");
      DEBUG("Task ",rank,": ","append_graph: src_start = ", src_start, " src_end = ", src_end, "\n");
      DEBUG("Task ",rank,": ","append_graph: total_num_nodes = ", total_num_nodes, "\n");

      // Create an I/O communicator
      MPI_Comm  io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      if (rank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);
      
      // A vector that maps nodes to compute ranks
      map< NODE_IDX_T, rank_t > node_rank_map;
      compute_node_rank_map(io_size, total_num_nodes, node_rank_map);


      // construct a map where each set of edges are arranged by destination I/O rank
      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };
      rank_edge_map_t rank_edge_map;
      for (auto iter: input_edge_map)
        {
          NODE_IDX_T dst = iter.first;
          // all source/destination node IDs must be in range
          assert(dst_start <= dst && dst < dst_end);
          edge_tuple_t& et        = iter.second;
          vector<NODE_IDX_T>& v   = get<0>(et);
          vector <AttrVal>& va    = get<1>(et);

          vector<NODE_IDX_T> adj_vector;
          for (auto & src: v)
            {
              if (!(src_start <= src && src <= src_end))
                {
                  printf("src = %u src_start = %lu src_end = %lu\n", src, src_start, src_end);
                }
              assert(src_start <= src && src <= src_end);
              adj_vector.push_back(src - src_start);
              num_edges++;
            }

          vector<size_t> p = sort_permutation(adj_vector, compare_nodes);

          apply_permutation_in_place(adj_vector, p);

          for (auto & a : va)
            {
              for (size_t i=0; i<a.float_values.size(); i++)
                {
                  apply_permutation_in_place(a.float_values[i], p);
                }
              for (size_t i=0; i<a.uint8_values.size(); i++)
                {
                  apply_permutation_in_place(a.uint8_values[i], p);
                }
              for (size_t i=0; i<a.uint16_values.size(); i++)
                {
                  apply_permutation_in_place(a.uint16_values[i], p);
                }
              for (size_t i=0; i<a.uint32_values.size(); i++)
                {
                  apply_permutation_in_place(a.uint32_values[i], p);
                }
              for (size_t i=0; i<a.int8_values.size(); i++)
                {
                  apply_permutation_in_place(a.int8_values[i], p);
                }
              for (size_t i=0; i<a.int16_values.size(); i++)
                {
                  apply_permutation_in_place(a.int16_values[i], p);
                }
              for (size_t i=0; i<a.int32_values.size(); i++)
                {
                  apply_permutation_in_place(a.int32_values[i], p);
                }
            }
          auto it = node_rank_map.find(dst);
          assert(it != node_rank_map.end());
          size_t dst_rank = it->second;
          edge_tuple_t& et1 = rank_edge_map[dst_rank][dst];
          vector<NODE_IDX_T> &src_vec = get<0>(et1);
          src_vec.insert(src_vec.end(),adj_vector.begin(),adj_vector.end());
          vector <AttrVal> &edge_attr_vec = get<1>(et1);

          size_t i=0;
          for (auto & edge_attr : edge_attr_vec)
            {
              AttrVal& a = va[i];
              edge_attr.float_values.resize(a.float_values.size());
              edge_attr.uint8_values.resize(a.uint8_values.size());
              edge_attr.uint16_values.resize(a.uint16_values.size());
              edge_attr.uint32_values.resize(a.uint32_values.size());
              edge_attr.int8_values.resize(a.int8_values.size());
              edge_attr.int16_values.resize(a.int16_values.size());
              edge_attr.int32_values.resize(a.int32_values.size());
              edge_attr.append(a);
              i++;
            }
        }



      // send buffer and structures for MPI Alltoall operation
      vector<char> sendbuf;
      vector<int> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);

      // Create MPI_PACKED object with the edges of vertices for the respective I/O rank
      size_t num_packed_edges = 0; 

      data::serialize_rank_edge_map (size, rank, rank_edge_map, num_packed_edges,
                                     sendcounts, sendbuf, sdispls);
      rank_edge_map.clear();
      DEBUG("Task ",rank,": ","append_graph: num_packed_edges = ", num_packed_edges, "\n");
      
      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks receive zero),
      //    and creates sendcounts and sdispls arrays
      
      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) == MPI_SUCCESS);
      
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
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) == MPI_SUCCESS);
      sendbuf.clear();
      sendcounts.clear();
      sdispls.clear();

      size_t num_unpacked_edges = 0, num_unpacked_nodes = 0;
      edge_map_t prj_edge_map;
      if (recvbuf_size > 0)
        {
          data::deserialize_rank_edge_map (size, recvbuf, recvcounts, rdispls, 
                                           prj_edge_map, num_unpacked_nodes, num_unpacked_edges);
        }
      DEBUG("Task ",rank,": ","append_graph: num_unpacked_edges = ", num_unpacked_edges, "\n");


      if (rank < io_size)
        {
          hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
          assert(fapl >= 0);
          assert(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
          
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
          assert(file >= 0);
          
          append_projection (file, src_pop_name, dst_pop_name,
                             src_start, src_end, dst_start, dst_end,
                             num_unpacked_edges, prj_edge_map,
                             edge_attr_names);
          
          assert(H5Fclose(file) >= 0);
          assert(H5Pclose(fapl) >= 0);
        }

      return 0;
    }
  }
}
