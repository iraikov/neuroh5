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
    template <class T>
    void pack_size_edge_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const model::EdgeAttr& edge_attr_values,
     int &sendsize
     )
    {
      for (size_t k = 0; k < edge_attr_values.size_attr_vec<T>(); k++)
        {
          int packsize=0;
          uint32_t numitems = edge_attr_values.size_attr<T>(k);
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
          sendsize += packsize;
          assert(MPI_Pack_size(numitems, mpi_type, comm, &packsize) ==
                 MPI_SUCCESS);
          sendsize += packsize;
        }
    }

    template <class T>
    void pack_edge_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const model::EdgeAttr& edge_attr_values,
     const int &sendbuf_size,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      for (size_t k = 0; k < edge_attr_values.size_attr_vec<T>(); k++)
        {
          uint32_t numitems = edge_attr_values.size_attr<T>(k);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          assert(MPI_Pack(edge_attr_values.attr_ptr<T>(k), numitems,
                          mpi_type, &sendbuf[0], sendbuf_size, &sendpos, comm)
                 == MPI_SUCCESS);
        }
    }

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

      pack_size_edge_attr_values<float>(comm, MPI_FLOAT, edge_attr_values,
                                        sendsize);
      pack_size_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, edge_attr_values,
                                        sendsize);
      pack_size_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, edge_attr_values,
                                           sendsize);
      pack_size_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, edge_attr_values,
                                           sendsize);
      sendbuf.resize(sendbuf.size() + sendsize);

      int sendbuf_size = sendbuf.size();
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

      pack_edge_attr_values<float>(comm, MPI_FLOAT, edge_attr_values,
                                   sendbuf_size, sendpos, sendbuf);
      pack_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, edge_attr_values,
                                     sendbuf_size, sendpos, sendbuf);
      pack_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, edge_attr_values,
                                      sendbuf_size, sendpos, sendbuf);
      pack_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, edge_attr_values,
                                      sendbuf_size, sendpos, sendbuf);

      return ierr;
    }


    /**************************************************************************
     * Unpack an MPI packed edge data structure into source vertices and edge
     * attributes
     **************************************************************************/

    template <class T>
    void unpack_edge_attr_values
    (
     MPI_Comm comm,
     MPI_Datatype mpi_type,
     const vector<size_t> &edge_attr_num,
     const size_t edge_attr_idx,
     const vector<uint8_t> &recvbuf,
     const int recvbuf_size,
     model::EdgeAttr& edge_attr_values,
     int & recvpos
     )
    {
      int ierr;
      for (size_t k = 0; k < edge_attr_num[edge_attr_idx]; k++)
        {
          uint32_t numitems=0;
          vector<T> vec;
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
          assert(ierr == MPI_SUCCESS);
          vec.resize(numitems);
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                            &vec[0], numitems, mpi_type,
                            comm);
          assert(ierr == MPI_SUCCESS);
          edge_attr_values.insert(vec);
        }

    }

    int unpack_edge
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     const vector<size_t> &edge_attr_num,
     NODE_IDX_T &dst,
     vector<NODE_IDX_T>& src_vector,
     model::EdgeAttr& edge_attr_values,
     int & recvpos
     )
    {
      int ierr = 0;
      uint32_t numitems = 0;
      const int recvbuf_size = recvbuf.size();

      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst, 1, NODE_IDX_MPI_T, comm);
      assert(ierr == MPI_SUCCESS);
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &numitems, 1, MPI_UINT32_T, comm);
      assert(ierr == MPI_SUCCESS);
      src_vector.resize(numitems);
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &src_vector[0], numitems, NODE_IDX_MPI_T,
                        comm);
      assert(ierr == MPI_SUCCESS);

      unpack_edge_attr_values<float>(comm, MPI_FLOAT, edge_attr_num, 0,
                                     recvbuf, recvbuf_size, 
                                     edge_attr_values, recvpos);
      unpack_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, edge_attr_num, 1,
                                       recvbuf, recvbuf_size, 
                                       edge_attr_values, recvpos);
      unpack_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, edge_attr_num, 2,
                                        recvbuf, recvbuf_size, 
                                        edge_attr_values, recvpos);
      unpack_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, edge_attr_num, 3,
                                        recvbuf, recvbuf_size, 
                                        edge_attr_values, recvpos);

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
          vector<NODE_IDX_T> send_edges, recv_edges, total_recv_edges;
          model::rank_edge_map_t prj_rank_edge_map;
          model::edge_map_t prj_edge_map;
          vector<size_t> edge_attr_num;
          size_t num_edges = 0, total_prj_num_edges = 0;

          DEBUG("projection ", i, "\n");

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

              DEBUG("projection ", i, " I/O\n");

              uint32_t dst_pop, src_pop;
              io::read_destination_population(io_comm, file_name, prj_names[i], dst_pop);
              io::read_source_population(io_comm, file_name, prj_names[i], src_pop);

              DEBUG("projection ", i, " after read_dest/read_source\n");

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
                  assert(io::hdf5::read_all_edge_attributes(io_comm, file_name, prj_names[i], 
                                                            edge_base, edge_count,
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
              printf("scatter: rank %d: packing edge data for projection %lu\n", rank, i);
              DEBUG("scatter: packing edge data from projection ", i, "(", prj_names[i], ")");
            
              for (size_t dst_rank = 0; dst_rank < (size_t)size; dst_rank++)
                {
                  auto it1 = prj_rank_edge_map.find(dst_rank); 
                  uint32_t rank_numitems=0;
                  sdispls[dst_rank] = sendpos;
                  if (it1 != prj_rank_edge_map.end())
                    {
                      rank_numitems = it1->second.size();
                    }

                  // Create MPI_PACKED object with the number of dst vertices for this rank
                  int packsize=0;
                  assert(MPI_Pack_size(1, MPI_UINT32_T, all_comm, &packsize) == MPI_SUCCESS);
                  sendbuf.resize(sendbuf.size() + packsize);
                  assert(MPI_Pack(&rank_numitems, 1, MPI_UINT32_T, &sendbuf[0], (int)sendbuf.size(),
                                  &sendpos, all_comm) == MPI_SUCCESS);
                  sendcounts[dst_rank] += packsize;

                  if (rank_numitems > 0)
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
              printf("rank %d: scatter: finished packing edges for projection %lu: num_edges = %lu num_packed_edges = %lu\n",
                     rank, i, num_edges, num_packed_edges);
              assert(num_packed_edges == num_edges);
              DEBUG("scatter: finished packing edge data from projection ", i, "(", prj_names[i], ")");
            }

          // 0. Broadcast the number of attributes of each type to all ranks
          printf("rank %d: scatter: before step 0\n", rank);
          edge_attr_num.resize(4);
          assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_UINT32_T, 0, all_comm) >= 0);
          printf("rank %d: scatter: after step 0\n", rank);

        
          // 1. Each ALL_COMM rank sends an edge vector size to
          //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
          //    and creates sendcounts and sdispls arrays
      
          assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) >= 0);
          printf("rank %d: scatter: after step 1\n", rank);
          DEBUG("scatter: after MPI_Alltoall sendcounts for projection ", i, "(", prj_names[i], ")");

          // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
          //    a receive buffer, recvcounts, and rdispls

          size_t recvbuf_size = recvcounts[0];
          for (int p = 1; p < size; ++p)
            {
              rdispls[p] = rdispls[p-1] + recvcounts[p-1];
              recvbuf_size += recvcounts[p];
            }
          printf("rank %d: scatter projection %lu: after step 2: recvbuf_size = %lu\n", rank, i, recvbuf_size);
          assert(recvbuf_size > 0);
          DEBUG("scatter: recvbuf_size = ", recvbuf_size);
          vector<uint8_t> recvbuf(recvbuf_size);

          // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
          assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                               &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                               all_comm) >= 0);
          printf("rank %d: scatter: after step 3\n", rank);
          sendbuf.clear();
          printf("scatter: rank %d after MPI_Alltoallv for projection %lu; recvbuf size = %lu\n", 
                 rank, i, recvbuf.size());

          int recvpos = 0; size_t num_recv_items=0;
          size_t num_recv_edges=0;
          while ((size_t)recvpos < recvbuf_size)
            {
              // Unpack number of received blocks for this rank
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, 
                                &recvpos, &num_recv_items, 1, MPI_UINT32_T, all_comm) ==
                     MPI_SUCCESS);
              
              printf("rank %d: scatter projection %lu: recvbuf_size = %lu num_recv_items = %lu\n",
                     rank, i, recvbuf_size, num_recv_items);
              
              for (size_t j = 0; j<num_recv_items; j++)
                {
                  NODE_IDX_T dst; 
                  vector<NODE_IDX_T> src_vector;
                  model::EdgeAttr edge_attr_values;
                  
                  unpack_edge(all_comm, recvbuf, edge_attr_num, 
                              dst, src_vector, edge_attr_values, recvpos);
                  num_recv_edges = num_recv_edges + src_vector.size();
                  if ((size_t)recvpos > recvbuf_size)
                    {
                      printf("rank %d: unpacking projection %lu has reached end of buffer; "
                             "recvpos = %d recvbuf_size = %lu\n", rank, i, recvpos, recvbuf_size);
                    }
                  assert((size_t)recvpos <= recvbuf_size);

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
            }
          printf("rank %d: scatter: after step 4 projection %lu: num_recv_edges = %lu recvpos = %d\n", rank, i, num_recv_edges, recvpos);
          recvbuf.clear();
          printf("rank %d: scatter: finished unpacking edges for projection %lu\n", rank, i);
          DEBUG("scatter: finished unpacking edges for projection ", i, "(", prj_names[i], ")");
          
          prj_vector.push_back(prj_edge_map);
          assert(MPI_Barrier(all_comm) == MPI_SUCCESS);
        }

      MPI_Comm_free(&io_comm);
      return ierr;
    }

  }
  
}
