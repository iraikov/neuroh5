// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_graph.cc
///
///  Top-level functions for appending edge information to graphs in
///  DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016-2025 Project NeuroH5.
//==============================================================================


#include "neuroh5_types.hh"
#include "alltoallv_template.hh"
#include "attr_map.hh"
#include "cell_populations.hh"
#include "append_graph.hh"
#include "append_projection.hh"
#include "edge_attributes.hh"
#include "path_names.hh"
#include "sort_permutation.hh"
#include "serialize_edge.hh"
#include "range_sample.hh"
#include "node_rank_map.hh"
#include "debug.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"

#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <cstdio>

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {


    
    int append_graph
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
      pop_label_map_t pop_labels;
      pop_range_map_t pop_ranges;
      size_t src_pop_idx, dst_pop_idx; bool src_pop_set=false, dst_pop_set=false;
      size_t total_num_nodes=0, pop_num_nodes=0;
      size_t dst_start, dst_end;
      size_t src_start, src_end;

      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };

      int ssize, srank; size_t size, rank;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &ssize) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &srank) == MPI_SUCCESS);
      throw_assert_nomsg(ssize > 0);
      throw_assert_nomsg(srank >= 0);
      size = ssize;
      rank = srank;

      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);

      if (ssize < io_size_arg)
        {
          io_size = size > 0 ? size : 1;
        }
      else
        {
          io_size = io_size_arg > 0 ? (size_t)io_size_arg : 1;
        }
      throw_assert_nomsg(cell::read_population_ranges(all_comm, file_name, pop_ranges, pop_num_nodes) >= 0);
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
      
      vector< NODE_IDX_T > local_node_index;
      for (auto iter: input_edge_map)
        {
          NODE_IDX_T dst = iter.first;
          local_node_index.push_back(dst);
        }
      
      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = (io_rank_set.find(rank) != io_rank_set.end());

      // Map nodes to compute ranks
      map< NODE_IDX_T, rank_t > node_rank_map;
      mpi::compute_node_rank_map(all_comm, io_rank_set, local_node_index,
                                 total_num_nodes, node_rank_map);
      if (total_num_nodes == 0)
        {
          throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
          return 0;
        }

      rank_edge_map_t rank_edge_map;
      mpi::MPI_DEBUG(all_comm, "append_graph: ", src_pop_name, " -> ", dst_pop_name, ": ",
                     " total_num_nodes = ", total_num_nodes);

      // construct a map where each set of edges are arranged by destination I/O rank
      for (auto iter: input_edge_map)
        {
          NODE_IDX_T dst          = iter.first;
          // all source/destination node IDs must be in range
          throw_assert_nomsg(dst_start <= dst && dst < dst_end);
          edge_tuple_t& et        = iter.second;
          const vector<NODE_IDX_T>& v   = get<0>(et);
          vector <AttrVal>& va    = get<1>(et);

          auto it = node_rank_map.find(dst);
          throw_assert_nomsg(it != node_rank_map.end());
          size_t dst_rank = it->second;
          edge_tuple_t& et1 = rank_edge_map[dst_rank][dst];

          if (v.size() > 0)
            {
              vector<NODE_IDX_T> adj_vector;
              for (const auto & src: v)
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

              vector<NODE_IDX_T> &src_vec = get<0>(et1);
              src_vec.insert(src_vec.end(),adj_vector.begin(),adj_vector.end());
              vector <AttrVal> &edge_attr_vec = get<1>(et1);
              edge_attr_vec.resize(va.size());
              
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
        }


      // send buffer and structures for MPI Alltoall operation
      vector<char> sendbuf, recvbuf;
      vector<size_t> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);

      // Create serialized object with the edges of vertices for the respective I/O rank
      size_t num_packed_edges = 0; 


      data::serialize_rank_edge_map (size, rank, rank_edge_map, num_packed_edges,
                                     sendcounts, sendbuf, sdispls);
      rank_edge_map.clear();

      throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                     recvcounts, rdispls, recvbuf) >= 0);
      sendbuf.clear();
      sendbuf.shrink_to_fit();
      sendcounts.clear();
      sdispls.clear();

      size_t num_unpacked_edges = 0, num_unpacked_nodes = 0;
      edge_map_t prj_edge_map;
      if (recvbuf.size() > 0)
        {
          data::deserialize_rank_edge_map (size, recvbuf, recvcounts, rdispls, 
                                           prj_edge_map, num_unpacked_nodes, num_unpacked_edges);
        }

      recvbuf.clear();
      recvcounts.clear();
      rdispls.clear();

      // Create an I/O communicator
      MPI_Comm  io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      if (is_io_rank)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }

      if (is_io_rank)
        {
          hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
          throw_assert_nomsg(fapl >= 0);
#ifdef HDF5_IS_PARALLEL
          throw_assert_nomsg(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
#endif
          
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
          throw_assert_nomsg(file >= 0);

          hdf5::create_projection_groups(file, src_pop_name, dst_pop_name);
          
          throw_assert_nomsg(H5Fclose(file) >= 0);
          
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
          throw_assert_nomsg(file >= 0);

          append_projection (io_comm, file, src_pop_name, dst_pop_name,
                             src_start, src_end, dst_start, dst_end,
                             num_unpacked_edges, prj_edge_map,
                             edge_attr_index, chunk_size);

          throw_assert_nomsg(MPI_Barrier(io_comm) == MPI_SUCCESS);
          throw_assert_nomsg(H5Fclose(file) >= 0);
          throw_assert_nomsg(H5Pclose(fapl) >= 0);
        }
      throw_assert_nomsg(MPI_Barrier(io_comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
      return 0;
    }
  }
}
