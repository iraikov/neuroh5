// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_reader.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "dbs_edge_reader.hh"

#include "attributes.hh"
#include "graph_reader.hh"
#include "population_reader.hh"
#include "read_graph.hh"
#include "read_population.hh"
#include "validate_edge_list.hh"

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

namespace ngh5
{
  namespace graph
  {
    /***************************************************************************
     * Prepare an MPI packed data structure with source vertices and edge
     * attributes for a given destination vertex.
     **************************************************************************/

    int pack_edge
    (
     MPI_Comm comm,
     const NODE_IDX_T &dst,
     const vector<NODE_IDX_T>& src_vector,
     const model::EdgeAttr& edge_attr_values,
     int &sendpos,
     int &sendsize,
     vector<uint8_t> &sendbuf
     )
    {
      int ierr = 0;
      int packsize;

      sendsize = 0;

      assert(MPI_Pack_size(1, NODE_IDX_MPI_T, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;

      assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      assert(MPI_Pack_size(src_vector.size(), NODE_IDX_MPI_T, comm, &packsize)
             == MPI_SUCCESS);
      sendsize += packsize;

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<float>(); k++)
        {
          size_t numitems = edge_attr_values.size_attr<float>(k);
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
          sendsize += packsize;
          assert(MPI_Pack_size(numitems, MPI_FLOAT, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint8_t>(); k++)
        {
          size_t numitems = edge_attr_values.size_attr<uint8_t>(k);
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
          assert(MPI_Pack_size(numitems, MPI_UNSIGNED_CHAR, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint16_t>(); k++)
        {
          size_t numitems = edge_attr_values.size_attr<uint16_t>(k);
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
          assert(MPI_Pack_size(numitems, MPI_UNSIGNED_SHORT, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint32_t>(); k++)
        {
          size_t numitems = edge_attr_values.size_attr<uint32_t>(k);
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
          assert(MPI_Pack_size(numitems, MPI_UNSIGNED, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
        }

      sendbuf.resize(sendbuf.size() + sendsize);

      size_t sendbuf_size = sendbuf.size();
      uint32_t numitems = 0;

      // Create MPI_PACKED object with all the source vertices and edge attributes
      assert(MPI_Pack(&dst, 1, NODE_IDX_MPI_T, &sendbuf[0], sendbuf_size,
                      &sendpos, comm) == MPI_SUCCESS);

      numitems = src_vector.size();
      assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size,
                      &sendpos, comm) == MPI_SUCCESS);
      assert(MPI_Pack(&src_vector[0], src_vector.size(), NODE_IDX_MPI_T,
                      &sendbuf[0], sendbuf_size, &sendpos, comm) ==
             MPI_SUCCESS);

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<float>(); k++)
        {
          numitems = edge_attr_values.size_attr<float>(k);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          assert(MPI_Pack(edge_attr_values.attr_ptr<float>(k), numitems,
                          MPI_FLOAT, &sendbuf[0], sendbuf_size, &sendpos, comm)
                 == MPI_SUCCESS);
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint8_t>(); k++)
        {
          numitems = edge_attr_values.size_attr<uint8_t>(k);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          assert(MPI_Pack(edge_attr_values.attr_ptr<uint8_t>(k), numitems,
                          MPI_UNSIGNED_CHAR, &sendbuf[0], sendbuf_size,
                          &sendpos, comm) == MPI_SUCCESS);
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint16_t>(); k++)
        {
          numitems = edge_attr_values.size_attr<uint16_t>(k);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          assert(MPI_Pack(edge_attr_values.attr_ptr<uint16_t>(k), numitems,
                          MPI_UNSIGNED_SHORT, &sendbuf[0], sendbuf_size,
                          &sendpos, comm) == MPI_SUCCESS);
        }

      for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint32_t>(); k++)
        {
          numitems = edge_attr_values.size_attr<uint32_t>(k);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          assert(MPI_Pack(edge_attr_values.attr_ptr<uint8_t>(k), numitems,
                          MPI_UNSIGNED_CHAR, &sendbuf[0], sendbuf_size,
                          &sendpos, comm) == MPI_SUCCESS);
        }

      return ierr;
    }


    /**************************************************************************
     * Unpack an MPI packed edge data structure into source vertices and edge
     * attributes
     **************************************************************************/

    int unpack_edge
    (
     MPI_Comm comm,
     NODE_IDX_T &dst,
     vector<NODE_IDX_T>& src_vector,
     model::EdgeAttr& edge_attr_values,
     int & recvpos,
     const vector<uint8_t> &recvbuf,
     const vector<size_t> &edge_attr_num
     )
    {
      int ierr = 0;
      uint32_t numitems;
      int recvbuf_size = recvbuf.size();

      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst, 1, NODE_IDX_MPI_T, comm) >= 0);

      MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
      src_vector.resize(numitems);
      MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                 &src_vector[0], numitems, NODE_IDX_MPI_T,
                 comm);

      for (size_t k = 0; k < edge_attr_num[0]; k++)
        {
          vector<float> vec;
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
          vec.resize(numitems);
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                     &vec[0], numitems, MPI_FLOAT,
                     comm);
          edge_attr_values.insert(vec);
        }

      for (size_t k = 0; k < edge_attr_num[1]; k++)
        {
          vector<uint8_t> vec;
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
          vec.resize(numitems);
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                     &(vec)[0], numitems, MPI_UNSIGNED_CHAR, comm);
          edge_attr_values.insert(vec);
        }

      for (size_t k = 0; k < edge_attr_num[2]; k++)
        {
          vector<uint16_t> vec;
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
          vec.resize(numitems);
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                     &(vec)[0], numitems, MPI_UNSIGNED_SHORT, comm);
          edge_attr_values.insert(vec);
        }

      for (size_t k = 0; k < edge_attr_num[3]; k++)
        {
          vector<uint32_t> vec;
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
          vec.resize(numitems);
          MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                     &(vec)[0], numitems, MPI_UNSIGNED, comm);
          edge_attr_values.insert(vec);
        }
  
  
      return ierr;
    }

    /*****************************************************************************
     * Load and scatter edge data structures 
     *****************************************************************************/

    int scatter_graph
    (
     MPI_Comm                      all_comm,
     const std::string&            file_name,
     const int                     io_size,
     const bool                    opt_attrs,
     const vector<string>&         prj_names,
     // A vector that maps nodes to compute ranks
     const vector<model::rank_t>&  node_rank_vector,
     vector < model::edge_map_t >& prj_vector,
     size_t                       &total_num_nodes,
     size_t                       &local_num_edges,
     size_t                       &total_num_edges
     )
    {
      int ierr = 0;
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      // The set of compute ranks for which the current I/O rank is responsible
      model::rank_edge_map_t rank_edge_map;
      set< pair<model::pop_t, model::pop_t> > pop_pairs;
      vector<model::pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,model::pop_t> > pop_ranges;
      uint64_t prj_size = 0;
  
      int rank, size;
      assert(MPI_Comm_size(all_comm, &size) >= 0);
      assert(MPI_Comm_rank(all_comm, &rank) >= 0);

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

  
      assert(MPI_Bcast(&prj_size, 1, MPI_UINT64_T, 0, all_comm) >= 0);
      DEBUG("rank ", rank, ": scatter: after bcast: prj_size = ", prj_size);

      // For each projection, I/O ranks read the edges and scatter
      for (size_t i = 0; i < prj_size; i++)
        {
          vector<uint8_t> sendbuf; int sendpos = 0;
          vector<int> sendcounts, sdispls, recvcounts, rdispls;
          vector<NODE_IDX_T> recv_edges, total_recv_edges;
          model::rank_edge_map_t prj_rank_edge_map;
          model::edge_map_t prj_edge_map;
          vector<size_t> edge_attr_num;
          size_t num_edges = 0, total_prj_num_edges = 0, num_recv_edges = 0;

          sendcounts.resize(size,0);
          sdispls.resize(size,0);
          recvcounts.resize(size,0);
          rdispls.resize(size,0);

          if (rank < io_size)
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
              io::read_destination_population(io_comm, file_name, prj_names[i], dst_pop);
              io::read_source_population(io_comm, file_name, prj_names[i], src_pop);

              dst_start = pop_vector[dst_pop].start;
              src_start = pop_vector[src_pop].start;

              DEBUG(" dst_start = ", dst_start,
                    " src_start = ", src_start,
                    "\n");

              DEBUG("scatter: reading projection ", i, "(", prj_names[i], ")");
              assert(io::hdf5::read_dbs_projection(io_comm, file_name, prj_names[i], 
                                                   dst_start, src_start, total_prj_num_edges,
                                                   block_base, edge_base, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
              DEBUG("scatter: validating projection ", i, "(", prj_names[i], ")");
              // validate the edges
              assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                        pop_ranges, pop_pairs) == true);


              if (opt_attrs)
                {
                  edge_count = src_idx.size();
                  assert(io::hdf5::get_edge_attributes(file_name, prj_names[i], edge_attr_info) >= 0);
                  assert(io::hdf5::num_edge_attributes(edge_attr_info, edge_attr_num) >= 0);
                  assert(io::hdf5::read_all_edge_attributes(io_comm, file_name, prj_names[i], edge_base, edge_count,
                                                  edge_attr_info, edge_attr_values) >= 0);
                }

              // append to the edge map
              assert(append_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                     edge_attr_values, node_rank_vector, num_edges, prj_rank_edge_map) >= 0);
      
              // ensure that all edges in the projection have been read and appended to edge_list
              assert(num_edges == src_idx.size());

            } // rank < io_size


          if (rank < io_size)
            {
              size_t num_packed_edges = 0;
              DEBUG("scatter: packing edge data from projection ", i, "(", prj_names[i], ")");
            
      
              for (auto it1 = prj_rank_edge_map.cbegin(); it1 != prj_rank_edge_map.cend(); ++it1)
                {
                  uint32_t dst_rank;
                  dst_rank = it1->first;
                  sdispls[dst_rank] = sendpos;
                  if (it1->second.size() > 0)
                    {
                      for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                        {
                          int sendsize;
                          NODE_IDX_T dst = it2->first;
                  
                          const vector<NODE_IDX_T>  src_vector = get<0>(it2->second);
                          const model::EdgeAttr&      my_edge_attrs = get<1>(it2->second);

                          num_packed_edges = num_packed_edges + src_vector.size();
                        
                          if (src_vector.size() > 0)
                            {
                      
                              assert(pack_edge(all_comm, dst, src_vector, my_edge_attrs,
                                               sendpos, sendsize, sendbuf) == 0);
                      
                              sendcounts[dst_rank] += sendsize;
                            }
                        }
                    }
          
                }
            
              // ensure the correct number of edges is being packed
              assert(num_packed_edges == num_edges);
            
              DEBUG("scatter: finished packing edge data from projection ", i, "(", prj_names[i], ")");
            }

          //MPI_Barrier(all_comm);
          
          // 0. Broadcast the number of attributes of each type to all ranks
          edge_attr_num.resize(4);
          assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_UINT32_T, 0, all_comm) >= 0);

        
          // 1. Each ALL_COMM rank sends an edge vector size to
          //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
          //    and creates sendcounts and sdispls arrays
      
          assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) >= 0);
          DEBUG("scatter: after MPI_Alltoall sendcounts for projection ", i, "(", prj_names[i], ")");

          // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
          //    a receive buffer, recvcounts, and rdispls

          size_t recvbuf_size = recvcounts[0];
          for (int p = 1; p < size; ++p)
            {
              rdispls[p] = rdispls[p-1] + recvcounts[p-1];
              recvbuf_size += recvcounts[p];
            }

          DEBUG("scatter: recvbuf_size = ", recvbuf_size);
          vector<uint8_t> recvbuf(recvbuf_size);
          int recvpos = 0;

          // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
          assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                               &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                               all_comm) >= 0);
          DEBUG("scatter: after MPI_Alltoallv for projection ", i, "(", prj_names[i], ")");

          while ((unsigned int)recvpos < recvbuf.size()-1)
            {
              NODE_IDX_T dst; 
              vector<NODE_IDX_T> src_vector;
              model::EdgeAttr           edge_attr_values;
          
              unpack_edge(all_comm, dst, src_vector, edge_attr_values, recvpos, recvbuf, edge_attr_num);
              num_recv_edges = num_recv_edges + src_vector.size();

              if (prj_edge_map.find(dst) == prj_edge_map.end())
                {
                  prj_edge_map.insert(make_pair(dst,make_tuple(src_vector, edge_attr_values)));
                }
              else
                {
                  model::edge_tuple_t et = prj_edge_map[dst];
                  vector<NODE_IDX_T> &v = get<0>(et);
                  model::EdgeAttr &a = get<1>(et);
                  v.insert(v.end(),src_vector.begin(),src_vector.end());
                  a.append(edge_attr_values);
                  prj_edge_map[dst] = make_tuple(v,a);
                }
            }
          DEBUG("scatter: finished unpacking edges for projection ", i, "(", prj_names[i], ")");

          prj_vector.push_back(prj_edge_map);
        }

      MPI_Comm_free(&io_comm);
      return ierr;
    }

  }
  
}
