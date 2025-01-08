// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_projection.cc
///
///  Top-level functions for reading edges in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2024 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection_datasets.hh"
#include "edge_attributes.hh"
#include "cell_populations.hh"
#include "validate_edge_list.hh"
#include "scatter_read_projection.hh"
#include "alltoallv_template.hh"
#include "serialize_edge.hh"
#include "serialize_data.hh"
#include "append_rank_edge_map.hh"
#include "range_sample.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    /*****************************************************************************
     * Load and scatter edge data structures 
     *****************************************************************************/

    int scatter_read_projection (MPI_Comm all_comm, const int io_size, EdgeMapType edge_map_type, 
                                 const string& file_name, const string& src_pop_name, const string& dst_pop_name, 
                                 const NODE_IDX_T& src_start,
                                 const NODE_IDX_T& dst_start,
                                 const vector<string> &attr_namespaces,
                                 const node_rank_map_t&  node_rank_map,
                                 const pop_search_range_map_t& pop_search_ranges,
                                 const set< pair<pop_t, pop_t> >& pop_pairs,
                                 vector < edge_map_t >& prj_vector,
                                 vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
                                 size_t &local_num_nodes, size_t &local_num_edges, size_t &total_num_edges,
                                 hsize_t& total_read_blocks,
                                 size_t offset, size_t numitems)
    {
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);

      throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);

      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = (io_rank_set.find(rank) != io_rank_set.end());

      size_t io_rank_root = 0;
      if (io_rank_set.size() > 0)
        {
          io_rank_root = *(io_rank_set.begin());
        }

      // Am I an I/O rank?
      if (is_io_rank)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }

      vector<NODE_IDX_T> send_edges, recv_edges, total_recv_edges;
      rank_edge_map_t prj_rank_edge_map;
      edge_map_t prj_edge_map;
      size_t num_edges = 0;
      map<string, vector< vector<string> > > edge_attr_names;
      
      local_num_nodes=0; local_num_edges=0;
      
      {
        vector<char> recvbuf;
        vector<size_t> recvcounts, rdispls;

        {
          vector<char> sendbuf; 
          vector<size_t> sendcounts(size,0), sdispls(size,0);

          mpi::MPI_DEBUG(all_comm, "scatter_read_projection: ", src_pop_name, " -> ", dst_pop_name, "\n");
          
          if (is_io_rank)
            {
              int io_rank;
              throw_assert_nomsg(MPI_Comm_rank(io_comm, &io_rank) == MPI_SUCCESS);


              DST_BLK_PTR_T block_base;
              DST_PTR_T edge_base, edge_count;
              vector<DST_BLK_PTR_T> dst_blk_ptr;
              vector<NODE_IDX_T> dst_idx;
              vector<DST_PTR_T> dst_ptr;
              vector<NODE_IDX_T> src_idx;
              map<string, data::NamedAttrVal> edge_attr_map;
              hsize_t local_read_blocks;

              mpi::MPI_DEBUG(io_comm, "scatter_read_projection: reading projection ", src_pop_name, " -> ", dst_pop_name);
              throw_assert_nomsg(hdf5::read_projection_datasets(io_comm, file_name, src_pop_name, dst_pop_name,
                                                                block_base, edge_base,
                                                                dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                                total_num_edges, total_read_blocks, local_read_blocks,
                                                                offset, numitems * size) >= 0);
          
              mpi::MPI_DEBUG(io_comm, "scatter_read_projection: validating projection ", src_pop_name, " -> ", dst_pop_name);
              // validate the edges
              throw_assert_nomsg(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                    pop_search_ranges, pop_pairs) == true);
          
              edge_count = src_idx.size();
              mpi::MPI_DEBUG(io_comm, "scatter_read_projection: reading attributes for ", src_pop_name, " -> ", dst_pop_name);
              for (const string& attr_namespace : attr_namespaces) 
                {
                  vector< pair<string,AttrKind> > edge_attr_info;
                  throw_assert_nomsg(graph::get_edge_attributes(io_comm, file_name, src_pop_name, dst_pop_name,
                                                                attr_namespace, edge_attr_info) >= 0);

                  throw_assert_nomsg(graph::read_all_edge_attributes(io_comm, file_name,
                                                                     src_pop_name, dst_pop_name, attr_namespace,
                                                                     edge_base, edge_count, edge_attr_info,
                                                                     edge_attr_map[attr_namespace]) >= 0);
                  
                  edge_attr_map[attr_namespace].attr_names(edge_attr_names[attr_namespace]);
                }

              
              // append to the edge map
              throw_assert_nomsg(data::append_rank_edge_map(rank, size, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                            attr_namespaces, edge_attr_map, node_rank_map, num_edges, prj_rank_edge_map,
                                                            edge_map_type) >= 0);
              
              mpi::MPI_DEBUG(io_comm, "scatter_read_projection: read ", num_edges,
                             " edges from projection ", src_pop_name, " -> ", dst_pop_name);
          
              // ensure that all edges in the projection have been read and appended to edge_list
              throw_assert(num_edges == src_idx.size(),
                           "edge count mismatch: num_edges = " << num_edges <<
                           " src_idx.size = " << src_idx.size());
                           
          
              size_t num_packed_edges = 0;
          
              data::serialize_rank_edge_map (size, rank, prj_rank_edge_map, 
                                             num_packed_edges, sendcounts, sendbuf, sdispls);

              // ensure the correct number of edges is being packed
              throw_assert_nomsg(num_packed_edges == num_edges);
              mpi::MPI_DEBUG(io_comm, "scatter_read_projection: packed ", num_packed_edges,
                        " edges from projection ", src_pop_name, " -> ", dst_pop_name);

            } // is_io_rank

          
          MPI_Comm_free(&io_comm);
          throw_assert_nomsg(MPI_Bcast(&total_read_blocks, 1, MPI_SIZE_T, io_rank_root, all_comm) == MPI_SUCCESS);
          throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                         recvcounts, rdispls, recvbuf) >= 0);
        }

        mpi::MPI_DEBUG(all_comm, "scatter_read_projection: recvbuf size is ", recvbuf.size());

        if (recvbuf.size() > 0)
          {
            data::deserialize_rank_edge_map (size, recvbuf, recvcounts, rdispls, 
                                             prj_edge_map, local_num_nodes, local_num_edges);
          }

        mpi::MPI_DEBUG(all_comm, "scatter_read_projection: prj_edge_map size is ", prj_edge_map.size());
        
        if (!attr_namespaces.empty())
          {
            vector<char> sendbuf; uint32_t sendbuf_size=0;
            if (rank == 0)
              {
                data::serialize_data(edge_attr_names, sendbuf);
                sendbuf_size = sendbuf.size();
              }

            throw_assert_nomsg(MPI_Barrier(all_comm) == MPI_SUCCESS);
            throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_UINT32_T, 0, all_comm) == MPI_SUCCESS);
            sendbuf.resize(sendbuf_size);
            throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, all_comm) == MPI_SUCCESS);
            
            if (rank != 0)
              {
                data::deserialize_data(sendbuf, edge_attr_names);
              }
            edge_attr_names_vector.push_back(edge_attr_names);
            
          }
      }

      
      mpi::MPI_DEBUG(all_comm, "scatter_read_projection: unpacked ", local_num_edges,
                     " edges for projection ", src_pop_name, " -> ", dst_pop_name);
      
      prj_vector.push_back(prj_edge_map);

      return 0;
    }


  }
  
}
