// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_reader.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "read_dbs_projection.hh"
#include "edge_attributes.hh"
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

    const int edge_start_delim = -1;
    const int edge_end_delim   = -2;
    const int rank_edge_start_delim = -3;
    const int rank_edge_end_delim   = -4;
    
    /***************************************************************************
     * Prepare an MPI packed data structure with source vertices and edge
     * attributes for a given destination vertex.
     **************************************************************************/
    template <class T>
    void pack_size_edge_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const size_t num_edges,
     const model::EdgeAttr& edge_attr_values,
     int &sendsize
     )
    {
      int packsize=0;
      size_t num_attrs = edge_attr_values.size_attr_vec<T>();
      if (num_attrs > 0)
        {
          for (size_t k = 0; k < num_attrs; k++)
            {
              uint32_t numitems = edge_attr_values.size_attr<T>(k);
              assert(MPI_Pack_size(numitems, mpi_type, comm, &packsize) == MPI_SUCCESS);
              sendsize += packsize;
            }
        }
    }

    template <class T>
    void pack_edge_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const uint32_t num_edges,
     const model::EdgeAttr& edge_attr_values,
     const int &sendbuf_size,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int status;
      vector<uint32_t> numitems_vector;
      uint32_t num_attrs = edge_attr_values.size_attr_vec<T>();
      if (num_attrs > 0)
        {
          assert(sendpos < sendbuf_size);
          for (size_t k = 0; k < num_attrs; k++)
            {
              uint32_t numitems = edge_attr_values.size_attr<T>(k);
              assert(num_edges == numitems);
              numitems_vector.push_back(numitems);
            }

          for (size_t k = 0; k < num_attrs; k++)
            {
              uint32_t numitems = edge_attr_values.size_attr<T>(k);
              status = MPI_Pack(&edge_attr_values.attr_vec<T>(k)[0], numitems,
                                mpi_type, &sendbuf[0], sendbuf_size, &sendpos, comm);
              assert(status == MPI_SUCCESS);
            }
          if (!(sendpos <= sendbuf_size))
            {
              printf("pack_edge_attr_values: sendpos = %d sendbuf_size = %u\n", sendpos, sendbuf_size);
            }
          assert(sendpos <= sendbuf_size);
        }
    }

    int pack_size_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const NODE_IDX_T &dst,
     const vector<NODE_IDX_T>& src_vector,
     const model::EdgeAttr& edge_attr_values,
     int &sendsize
     )
    {
      int ierr = 0;
      int packsize = 0;
      uint32_t num_edges = src_vector.size();

      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

          
#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack_size(2, MPI_INT, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
#endif
      
      assert(MPI_Pack_size(1, header_type, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      if (num_edges > 0)
        {
          assert(MPI_Pack_size(num_edges, NODE_IDX_MPI_T, comm, &packsize)
                 == MPI_SUCCESS);
          sendsize += packsize;

          pack_size_edge_attr_values<float>(comm, MPI_FLOAT, num_edges, 
                                            edge_attr_values, sendsize);
          pack_size_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, num_edges,
                                              edge_attr_values, sendsize);
          pack_size_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, num_edges,
                                               edge_attr_values, sendsize);
          pack_size_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, num_edges,
                                               edge_attr_values, sendsize);

        }

      sendsize+=MPI_BSEND_OVERHEAD;

      return ierr;

    }

    int pack_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const size_t dst_rank,
     const NODE_IDX_T dst,
     const vector<NODE_IDX_T>& src_vector,
     const model::EdgeAttr& edge_attr_values,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int ierr = 0;

      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      int sendbuf_size = sendbuf.size();
      assert(sendpos < sendbuf_size);
      assert(src_vector.size() > 0);

                
#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&edge_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif

      // Create MPI_PACKED object with all the source vertices and edge attributes
      size_t num_edges = src_vector.size();
      EdgeHeader header;
      
      header.dst = dst;
      header.size = num_edges;

      assert(MPI_Pack(&header, 1, header_type, &sendbuf[0], sendbuf_size,
                      &sendpos, comm) == MPI_SUCCESS);

      if (num_edges > 0)
        {
          assert(MPI_Pack(&src_vector[0], num_edges, NODE_IDX_MPI_T,
                          &sendbuf[0], sendbuf_size, &sendpos, comm) ==
                 MPI_SUCCESS);

          pack_edge_attr_values<float>(comm, MPI_FLOAT, num_edges, edge_attr_values,
                                       sendbuf_size, sendpos, sendbuf);
          pack_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, num_edges, edge_attr_values, 
                                         sendbuf_size, sendpos, sendbuf);
          pack_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, num_edges, edge_attr_values, 
                                          sendbuf_size, sendpos, sendbuf);
          pack_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, num_edges, edge_attr_values, 
                                          sendbuf_size, sendpos, sendbuf);
        }
                
#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&edge_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif

      return ierr;

    }

    
    int pack_edge_map
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     MPI_Datatype size_type,
     const size_t dst_rank,
     const model::edge_map_t& edge_map,
     size_t &num_packed_edges,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int ierr=0;
      // Create MPI_PACKED object with the number of dst vertices for this rank
      int packsize=0, sendsize=0;
      uint32_t rank_numitems=edge_map.size();
      assert(MPI_Pack_size(1, size_type, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      if (rank_numitems > 0)
        {

          for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
            {
              NODE_IDX_T dst = it->first;
              
              const vector<NODE_IDX_T>  src_vector = get<0>(it->second);
              const model::EdgeAttr&      my_edge_attrs = get<1>(it->second);
              
              num_packed_edges += src_vector.size();
              
              ierr = pack_size_edge(comm, header_type, dst, src_vector, my_edge_attrs,
                                    sendsize);
              assert(ierr == 0);
            }
        }
      sendbuf.resize(sendbuf.size() + sendsize, 0);
          
      Size sizeval;
      sizeval.size = rank_numitems;

      assert(MPI_Pack(&sizeval, 1, size_type, &sendbuf[0],
                      (int)sendbuf.size(), &sendpos, comm) == MPI_SUCCESS);
      if (rank_numitems > 0)
        {
          for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
            {
              NODE_IDX_T dst = it->first;
              
              const vector<NODE_IDX_T>  src_vector = get<0>(it->second);
              const model::EdgeAttr&      my_edge_attrs = get<1>(it->second);

              ierr = pack_edge(comm, header_type, dst_rank, dst, src_vector, my_edge_attrs,
                               sendpos, sendbuf);
              assert(ierr == 0);
              assert((size_t)sendpos <= sendbuf.size());
            }
        }

      return ierr;
    }

    
    void pack_rank_edge_map (MPI_Comm comm, MPI_Datatype header_type, MPI_Datatype size_type,
                             model::rank_edge_map_t& prj_rank_edge_map,
                             size_t &num_packed_edges,
                             vector<int>& sendcounts,
                             vector<uint8_t> &sendbuf,
                             vector<int> &sdispls
                             )
    {
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      int sendpos = 0;
      for (size_t dst_rank = 0; (int)dst_rank < size; dst_rank++)
        {
          sdispls[dst_rank] = sendpos;
          
#ifdef USE_EDGE_DELIM      
      int packsize=0;
      assert(MPI_Pack_size(2, MPI_INT, comm, &packsize) == MPI_SUCCESS);
      sendbuf.resize(sendbuf.size() + packsize);
      assert(MPI_Pack(&rank_edge_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif
          auto it1 = prj_rank_edge_map.find(dst_rank); 
          pack_edge_map (comm, header_type, size_type, dst_rank, 
                         it1->second, num_packed_edges, sendpos, sendbuf);

#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&rank_edge_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif
          assert(sendpos <= sendbuf.size());
          sendcounts[dst_rank] = sendpos - sdispls[dst_rank];
        }
      
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
     const uint32_t num_edges,
     const uint32_t edge_attr_num,
     const vector<uint8_t> &recvbuf,
     const int recvbuf_size,
     model::EdgeAttr& edge_attr_values,
     int & recvpos
     )
    {
      
      int ierr;
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      if (edge_attr_num > 0)
        {
          if (!(recvpos < recvbuf_size))
            {
              printf("recvbuf_size = %u recvpos = %d\n", recvbuf_size, recvpos);
            }
          assert(recvpos < recvbuf_size);

          for (size_t k = 0; k < edge_attr_num; k++)
            {
              assert(recvpos < recvbuf_size);
              vector<T> vec;
              vec.resize(num_edges);
              ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                                &vec[0], num_edges, mpi_type, comm);
              assert(ierr == MPI_SUCCESS);
              edge_attr_values.insert(vec);
            }
        }

    }

    int unpack_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const vector<uint8_t> &recvbuf,
     const vector<uint32_t> &edge_attr_num,
     NODE_IDX_T &dst,
     vector<NODE_IDX_T>& src_vector,
     model::EdgeAttr& edge_attr_values,
     int & recvpos
     )
    {
      int ierr = 0;
      const int recvbuf_size = recvbuf.size();
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      assert(recvpos < recvbuf_size);

#ifdef USE_EDGE_DELIM
      int delim=0;
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm);
      assert(ierr == MPI_SUCCESS);
      if (delim != -1)
        {
          printf("rank %d: unpack_edge: recvpos = %d recvbuf_size = %u delim = %d\n", 
                 rank, recvpos, recvbuf_size, delim);
        }
      assert(delim == edge_start_delim);
#endif


      EdgeHeader header;
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &header, 1, header_type, comm);
      assert(ierr == MPI_SUCCESS);
      dst = header.dst;
      size_t numitems = header.size;
      if (numitems > 0)
        {
          src_vector.resize(numitems,0);
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                            &src_vector[0], numitems, NODE_IDX_MPI_T,
                            comm);
          assert(ierr == MPI_SUCCESS);

          
          if (!(recvpos <= recvbuf_size))
            {
              printf("rank %d: unpack_edge: recvbuf_size = %u recvpos = %d dst = %u numitems = %u\n", 
                     rank, recvbuf_size, recvpos, dst, numitems);
            }
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<float>(comm, MPI_FLOAT, numitems, edge_attr_num[0],
                                         recvbuf, recvbuf_size, 
                                         edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, numitems, edge_attr_num[1],
                                           recvbuf, recvbuf_size, 
                                           edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, numitems, edge_attr_num[2],
                                            recvbuf, recvbuf_size, 
                                            edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, numitems, edge_attr_num[3],
                                            recvbuf, recvbuf_size, 
                                            edge_attr_values, recvpos);

        }

#ifdef USE_EDGE_DELIM
      delim=0;
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm);
      assert(ierr == MPI_SUCCESS);
      if (delim != edge_end_delim)
        {
          printf("rank %d: unpack_edge: recvpos = %d recvbuf_size = %u delim = %d\n", 
                 rank, recvpos, recvbuf_size, delim);
        }
      assert(delim == edge_end_delim);
#endif

      return ierr;
    }


    void unpack_rank_edge_map (MPI_Comm comm,
                               MPI_Datatype header_type,
                               MPI_Datatype size_type,
                               const size_t io_size,
                               const vector<uint8_t> &recvbuf,
                               const vector<int>& recvcounts,
                               const vector<int>& rdispls,
                               const vector<uint32_t> &edge_attr_num,
                               model::edge_map_t& prj_edge_map
                              )
    {
      const int recvbuf_size = recvbuf.size();
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      for (size_t ridx = 0; (int)ridx < size; ridx++)
        {
          if (recvcounts[ridx] > 0)
            {
              int recvpos = rdispls[ridx];
              assert(recvpos < recvbuf_size);
#ifdef USE_EDGE_DELIM
              int delim=0;
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
              if (delim != rank_edge_start_delim)
                {
                  printf("rank %d: unpack_rank_edge_map: ridx = %u recvcounts[%u] = %d recvpos = %d recvbuf_size = %u delim = %d\n", 
                         rank, ridx, ridx, recvcounts[ridx], recvpos, recvbuf_size, delim);
                }
              assert(delim == rank_edge_start_delim);
#endif
          
              Size sizeval;
              size_t num_recv_items=0; size_t num_recv_edges=0;
              
              // Unpack number of received blocks for this rank
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, 
                                &recvpos, &sizeval, 1, size_type, comm) ==
                     MPI_SUCCESS);
              num_recv_items = sizeval.size;
              if (num_recv_items > 0)
                {
                  for (size_t j = 0; j<num_recv_items; j++)
                    {
                      NODE_IDX_T dst; 
                      vector<NODE_IDX_T> src_vector;
                      model::EdgeAttr edge_attr_values;
                      
                      unpack_edge(comm, header_type, recvbuf, edge_attr_num, 
                                  dst, src_vector, edge_attr_values, recvpos);
                      num_recv_edges = src_vector.size();
                      if ((size_t)recvpos > recvbuf_size)
                        {
                          printf("rank %d: unpacking projection has reached end of buffer; "
                                 "recvpos = %d recvbuf_size = %lu j = %lu num_recv_items = %lu\n", 
                                 rank, recvpos, recvbuf_size, j, num_recv_items);
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

#ifdef USE_EDGE_DELIM
              delim=0;
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
              if (delim != rank_edge_end_delim)
                {
                  printf("rank %d: unpack_rank_edge_map: recvpos = %d recvbuf_size = %u delim = %d\n", 
                         rank, recvpos, recvbuf_size, delim);
                }
              assert(delim == rank_edge_end_delim);
#endif

            }
        }
    }

    
    /*****************************************************************************
     * Load and scatter edge data structures 
     *****************************************************************************/

    int scatter_projection (MPI_Comm all_comm, MPI_Comm io_comm, const int io_size,
                            MPI_Datatype header_type, MPI_Datatype size_type, 
                            const string& file_name, const string& prj_name, 
                            const bool opt_attrs,
                            const vector<model::rank_t>&  node_rank_vector,
                            const vector<model::pop_range_t>& pop_vector,
                            const map<NODE_IDX_T,pair<uint32_t,model::pop_t> >& pop_ranges,
                            const set< pair<model::pop_t, model::pop_t> >& pop_pairs,
                            vector < model::edge_map_t >& prj_vector
                            )
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
                                               block_base, edge_base, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
          
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
          
          assert(append_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                 edge_attr_values, node_rank_vector, num_edges, prj_rank_edge_map) >= 0);
          
          // ensure that all edges in the projection have been read and appended to edge_list
          assert(num_edges == src_idx.size());
          
          size_t num_packed_edges = 0;
          DEBUG("scatter: packing edge data from projection ", prj_name);
          
          pack_rank_edge_map (all_comm, header_type, size_type, prj_rank_edge_map, num_packed_edges, 
                              sendcounts, sendbuf, sdispls);

          prj_rank_edge_map.clear();
          
          // ensure the correct number of edges is being packed
          assert(num_packed_edges == num_edges);
          DEBUG("scatter: finished packing edge data from projection ", prj_name);
        } // rank < io_size
    
      // 0. Broadcast the number of attributes of each type to all ranks
      edge_attr_num.resize(4);
      assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_UINT32_T, 0, all_comm) == MPI_SUCCESS);
      
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
      assert(recvbuf_size > 0);

      vector<uint8_t> recvbuf;
      recvbuf.resize(recvbuf_size, 0);
      
      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) == MPI_SUCCESS);
      sendbuf.clear();
      
      unpack_rank_edge_map (all_comm, header_type, size_type, io_size, recvbuf, recvcounts, rdispls, edge_attr_num, prj_edge_map);
      
      DEBUG("scatter: finished unpacking edges for projection ", prj_name);
      
      prj_vector.push_back(prj_edge_map);
      assert(MPI_Barrier(all_comm) == MPI_SUCCESS);

      return 0;
    }


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
      
      header_disp[0] = reinterpret_cast<const unsigned char*>(&header.dst) - 
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
          scatter_projection(all_comm, io_comm, io_size, header_type, size_type, file_name, prj_names[i],
                             opt_attrs, node_rank_vector, pop_vector, pop_ranges, pop_pairs,
                             prj_vector);
                             
        }
      MPI_Comm_free(&io_comm);
      MPI_Type_free(&header_type);
      MPI_Type_free(&size_type);
      return ierr;
    }

  }
  
}
