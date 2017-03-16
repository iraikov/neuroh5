// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "read_dbs_projection.hh"
#include "edge_attributes.hh"
#include "population_reader.hh"
#include "read_population.hh"
#include "validate_edge_list.hh"
#include "scatter_graph.hh"
#include "bcast_string_vector.hh"
#include "pack_edge.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace ngh5;

#define MAX_ATTR_NAME 1024

namespace ngh5
{
  namespace graph
  {


    
    /*****************************************************************************
     * Load and scatter edge data structures 
     *****************************************************************************/

    int scatter_projection (MPI_Comm all_comm, MPI_Comm io_comm, const int io_size,
                            EdgeMapType edge_map_type, MPI_Datatype header_type, MPI_Datatype size_type, 
                            const string& file_name, const string& prj_name, 
                            const bool opt_attrs,
                            const vector<model::rank_t>&  node_rank_vector,
                            const vector<model::pop_range_t>& pop_vector,
                            const map<NODE_IDX_T,pair<uint32_t,model::pop_t> >& pop_ranges,
                            const set< pair<model::pop_t, model::pop_t> >& pop_pairs,
                            vector < model::edge_map_t >& prj_vector,
                            vector<vector<vector<string>>>& edge_attr_names_vector)
    {

      int rank, size;
      assert(MPI_Comm_size(all_comm, &size) >= 0);
      assert(MPI_Comm_rank(all_comm, &rank) >= 0);

      vector<uint8_t> sendbuf; 
      vector<int> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);
      vector<NODE_IDX_T> send_edges, recv_edges, total_recv_edges;
      model::rank_edge_map_t prj_rank_edge_map;
      model::edge_map_t prj_edge_map;
      vector<uint32_t> edge_attr_num;
      vector<vector<string>> edge_attr_names;
      size_t num_edges = 0, total_prj_num_edges = 0;
      
      DEBUG("projection ", prj_name, "\n");


      if (rank < (int)io_size)
        {
          DST_BLK_PTR_T block_base;
          DST_PTR_T edge_base, edge_count;
          NODE_IDX_T dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;
          vector< pair<string,hid_t> > edge_attr_info;
          model::EdgeNamedAttr edge_attr_values;
          
          uint32_t dst_pop, src_pop;
          io::read_destination_population(io_comm, file_name, prj_name, dst_pop);
          io::read_source_population(io_comm, file_name, prj_name, src_pop);
          
          DEBUG("projection ", prj_name, " after read_dest/read_source\n");
          
          dst_start = pop_vector[dst_pop].start;
          src_start = pop_vector[src_pop].start;
          

          DEBUG("scatter: reading projection ", prj_name);
          assert(io::hdf5::read_dbs_projection(io_comm, file_name, prj_name, 
                                               dst_start, src_start, total_prj_num_edges,
                                               block_base, edge_base, dst_blk_ptr, dst_idx,
                                               dst_ptr, src_idx) >= 0);
          
          DEBUG("scatter: validating projection ", prj_name);
          // validate the edges
          assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                    pop_ranges, pop_pairs) == true);
          
          
          if (opt_attrs)
            {
              edge_count = src_idx.size();
              assert(io::hdf5::get_edge_attributes(file_name, prj_name, edge_attr_info) >= 0);
              assert(io::hdf5::num_edge_attributes(edge_attr_info, edge_attr_num) >= 0);
              assert(io::hdf5::read_all_edge_attributes(io_comm, file_name, prj_name, 
                                                        edge_base, edge_count,
                                                        edge_attr_info, edge_attr_values) >= 0);
            }

          // append to the edge map
          
          assert(append_rank_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                      edge_attr_values, node_rank_vector, num_edges, prj_rank_edge_map,
                                      edge_map_type) >= 0);

          edge_attr_values.attr_names(edge_attr_names);
          
          // ensure that all edges in the projection have been read and appended to edge_list
          assert(num_edges == src_idx.size());
          
          size_t num_packed_edges = 0;
          DEBUG("scatter: packing edge data from projection ", prj_name);
          
          mpi::pack_rank_edge_map (all_comm, header_type, size_type, prj_rank_edge_map, 
                                   num_packed_edges, sendcounts, sendbuf, sdispls);

          // ensure the correct number of edges is being packed
          assert(num_packed_edges == num_edges);
          DEBUG("scatter: finished packing edge data from projection ", prj_name);
        } // rank < io_size
    
      // 0. Broadcast the number and names of attributes of each type to all ranks
      edge_attr_num.resize(4);
      assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_UINT32_T, 0, all_comm) == MPI_SUCCESS);
      edge_attr_names.resize(4);
      for (size_t aidx=0; aidx<edge_attr_names.size(); aidx++)
        {
          if (edge_attr_num[aidx] > 0)
            {
              assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME, edge_attr_names[aidx]) == MPI_SUCCESS);
            }
        }
      
      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
      //    and creates sendcounts and sdispls arrays
      
      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) == MPI_SUCCESS);
      DEBUG("scatter: after MPI_Alltoall sendcounts for projection ", prj_name);
      
      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      
      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < size; p++)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      vector<uint8_t> recvbuf;
      recvbuf.resize(recvbuf_size > 0 ? recvbuf_size : 1, 0);
      
      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) == MPI_SUCCESS);
      sendbuf.clear();
      sendcounts.clear();
      sdispls.clear();

      if (recvbuf_size > 0)
        {
          mpi::unpack_rank_edge_map (all_comm, header_type, size_type, io_size,
                                     recvbuf, recvcounts, rdispls, edge_attr_num,
                                     prj_edge_map);
        }
      
      DEBUG("scatter: finished unpacking edges for projection ", prj_name);
      
      prj_vector.push_back(prj_edge_map);
      edge_attr_names_vector.push_back(edge_attr_names);
      
      assert(MPI_Barrier(all_comm) == MPI_SUCCESS);

      return 0;
    }


    int scatter_graph
    (
     MPI_Comm                      all_comm,
     const EdgeMapType             edge_map_type,
     const std::string&            file_name,
     const int                     io_size,
     const bool                    opt_attrs,
     const vector<string>&         prj_names,
     // A vector that maps nodes to compute ranks
     const vector<model::rank_t>&  node_rank_vector,
     vector < model::edge_map_t >& prj_vector,
     vector < vector <vector<string>> >& edge_attr_names_vector,
     size_t                       &total_num_nodes,
     size_t                       &local_num_edges,
     size_t                       &total_num_edges
     )
    {
      int ierr = 0;
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI datatype for edges
      MPI_Datatype mpi_edge_type;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      // The set of compute ranks for which the current I/O rank is responsible
      set< pair<model::pop_t, model::pop_t> > pop_pairs;
      vector<model::pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,model::pop_t> > pop_ranges;
      uint64_t prj_size = 0;
      
      
      int rank, size;
      assert(MPI_Comm_size(all_comm, &size) >= 0);
      assert(MPI_Comm_rank(all_comm, &rank) >= 0);

      // Create an MPI datatype to describe the sizes of edge structures
      Size sizeval;
      MPI_Datatype size_type, size_struct_type;
      MPI_Datatype size_fld_types[1] = { MPI_UINT32_T };
      int size_blocklen[1] = { 1 };
      MPI_Aint size_disp[1];
      
      size_disp[0] = reinterpret_cast<const unsigned char*>(&sizeval.size) - 
        reinterpret_cast<const unsigned char*>(&sizeval);
      assert(MPI_Type_create_struct(1, size_blocklen, size_disp, size_fld_types, &size_struct_type) == MPI_SUCCESS);
      assert(MPI_Type_create_resized(size_struct_type, 0, sizeof(sizeval), &size_type) == MPI_SUCCESS);
      assert(MPI_Type_commit(&size_type) == MPI_SUCCESS);
      
      EdgeHeader header;
      MPI_Datatype header_type, header_struct_type;
      MPI_Datatype header_fld_types[2] = { NODE_IDX_MPI_T, MPI_UINT32_T };
      int header_blocklen[2] = { 1, 1 };
      MPI_Aint header_disp[2];
      
      header_disp[0] = reinterpret_cast<const unsigned char*>(&header.key) - 
        reinterpret_cast<const unsigned char*>(&header);
      header_disp[1] = reinterpret_cast<const unsigned char*>(&header.size) - 
        reinterpret_cast<const unsigned char*>(&header);
      assert(MPI_Type_create_struct(2, header_blocklen, header_disp, header_fld_types, &header_struct_type) == MPI_SUCCESS);
      assert(MPI_Type_create_resized(header_struct_type, 0, sizeof(header), &header_type) == MPI_SUCCESS);
      assert(MPI_Type_commit(&header_type) == MPI_SUCCESS);
      
      // Am I an I/O rank?
      if (rank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
      
          // read the population info
          assert(io::hdf5::read_population_combos(io_comm, file_name, pop_pairs)
                 >= 0);
          assert(io::hdf5::read_population_ranges
                 (io_comm, file_name, pop_ranges, pop_vector, total_num_nodes)
                 >= 0);
          prj_size = prj_names.size();
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);
  
      assert(MPI_Bcast(&prj_size, 1, MPI_UINT64_T, 0, all_comm) == MPI_SUCCESS);
      DEBUG("rank ", rank, ": scatter: after bcast: prj_size = ", prj_size);

      // For each projection, I/O ranks read the edges and scatter
      for (size_t i = 0; i < prj_size; i++)
        {
          scatter_projection(all_comm, io_comm, io_size, edge_map_type, header_type, size_type, file_name, prj_names[i],
                             opt_attrs, node_rank_vector, pop_vector, pop_ranges, pop_pairs,
                             prj_vector, edge_attr_names_vector);
                             
        }
      MPI_Comm_free(&io_comm);
      MPI_Type_free(&header_type);
      MPI_Type_free(&size_type);
      return ierr;
    }

  }
  
}
