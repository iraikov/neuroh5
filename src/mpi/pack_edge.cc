// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file pack_edge.cc
///
///  Top-level functions for packing/unpacking graphs edges in MPI_PACKED format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "pack_edge.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#include "cereal/archives/portable_binary.hpp"

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace mpi
  {
    
    const int edge_start_delim = -1;
    const int edge_end_delim   = -2;
    const int edge_attr_start_delim = -3;
    const int edge_attr_end_delim   = -4;
    const int rank_edge_start_delim = -5;
    const int rank_edge_end_delim   = -6;
    
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
     const data::AttrVal& edge_attr_values,
     int &sendsize
     )
    {
      int packsize=0;

      size_t num_attrs = edge_attr_values.size_attr_vec<T>();
      if (num_attrs > 0)
        {
          
#ifdef USE_EDGE_DELIM      
          assert(MPI_Pack_size(2, MPI_INT, comm, &packsize) == MPI_SUCCESS);
          sendsize += packsize;
#endif
          for (size_t k = 0; k < num_attrs; k++)
            {
              uint32_t numitems = edge_attr_values.size_attr<T>(k);
              assert(numitems == num_edges);
              assert(MPI_Pack_size( numitems, mpi_type, comm, &packsize) == MPI_SUCCESS);
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
     const data::AttrVal& edge_attr_values,
     const int &sendbuf_size,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int status;
      uint32_t num_attrs = edge_attr_values.size_attr_vec<T>();
      if (num_attrs > 0)
        {
          assert(sendpos < sendbuf_size);

#ifdef USE_EDGE_DELIM      
          assert(MPI_Pack(&edge_attr_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                          &sendpos, comm) == MPI_SUCCESS);

#endif

          for (size_t k = 0; k < num_attrs; k++)
            {
              uint32_t numitems = edge_attr_values.size_attr<T>(k);
              assert(numitems == num_edges);
              status = MPI_Pack(&edge_attr_values.attr_vec<T>(k)[0], numitems,
                                mpi_type, &sendbuf[0], sendbuf_size, &sendpos, comm);
              assert(status == MPI_SUCCESS);
            }
          if (!(sendpos <= sendbuf_size))
            {
              printf("pack_edge_attr_values: sendpos = %d sendbuf_size = %u\n", sendpos, sendbuf_size);
            }
          assert(sendpos <= sendbuf_size);

#ifdef USE_EDGE_DELIM      
          assert(MPI_Pack(&edge_attr_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                          &sendpos, comm) == MPI_SUCCESS);
          
#endif
        }

    }

    int pack_size_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const NODE_IDX_T &key_node,
     const vector<NODE_IDX_T>& adj_vector,
     const data::AttrVal& edge_attr_values,
     int &sendsize
     )
    {
      int ierr = 0;
      int packsize = 0;
      uint32_t num_edges = adj_vector.size();

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
          pack_size_edge_attr_values<int8_t>(comm, MPI_INT8_T, num_edges,
                                              edge_attr_values, sendsize);
          pack_size_edge_attr_values<int16_t>(comm, MPI_INT16_T, num_edges,
                                              edge_attr_values, sendsize);
          pack_size_edge_attr_values<int32_t>(comm, MPI_INT32_T, num_edges,
                                              edge_attr_values, sendsize);

        }

      return ierr;

    }

    int pack_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const NODE_IDX_T key_node,
     const vector<NODE_IDX_T>& adj_vector,
     const data::AttrVal& edge_attr_values,
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
                
#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&edge_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif

      // Create MPI_PACKED object with all the source vertices and edge attributes
      size_t num_edges = adj_vector.size();
      EdgeHeader header;
      
      header.key  = key_node;
      header.size = num_edges;

      assert(MPI_Pack(&header, 1, header_type, &sendbuf[0], sendbuf_size,
                      &sendpos, comm) == MPI_SUCCESS);

      
      if (num_edges > 0)
        {
          assert(MPI_Pack(&adj_vector[0], num_edges, NODE_IDX_MPI_T,
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
          pack_edge_attr_values<int8_t>(comm, MPI_INT8_T, num_edges, edge_attr_values, 
                                        sendbuf_size, sendpos, sendbuf);
          pack_edge_attr_values<int16_t>(comm, MPI_INT16_T, num_edges, edge_attr_values, 
                                         sendbuf_size, sendpos, sendbuf);
          pack_edge_attr_values<int32_t>(comm, MPI_INT32_T, num_edges, edge_attr_values, 
                                         sendbuf_size, sendpos, sendbuf);
        }
                
#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&edge_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif

      return ierr;

    }

    
    int pack_edge_map1 (MPI_Comm comm,
                        MPI_Datatype header_type,
                        MPI_Datatype size_type,
                        const edge_map_t& edge_map,
                        size_t &num_packed_edges,
                        int &sendpos,
                        vector<uint8_t> &sendbuf
                       )
    {
      int ierr=0;
      // Create MPI_PACKED object with the number of dst vertices for this rank
      int packsize=0, sendsize=0;

      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);


      uint32_t numitems=edge_map.size();

      assert(MPI_Pack_size(1, size_type, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      if (numitems > 0)
        {

          for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
            {
              NODE_IDX_T key_node = it->first;
              
              const vector<NODE_IDX_T>&  adj_vector = get<0>(it->second);
              const data::AttrVal&    my_edge_attrs = get<1>(it->second);

              num_packed_edges += adj_vector.size();
              
              ierr = pack_size_edge(comm, header_type, key_node, adj_vector, my_edge_attrs,
                                    sendsize);
              assert(ierr == 0);
            }
        }
      sendbuf.resize(sendbuf.size() + sendsize, 0);
          
      Size sizeval;
      sizeval.size = numitems;

      assert(MPI_Pack(&sizeval, 1, size_type, &sendbuf[0],
                      (int)sendbuf.size(), &sendpos, comm) == MPI_SUCCESS);
      if (numitems > 0)
        {
          for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
            {
              NODE_IDX_T key_node = it->first;
              
              const vector<NODE_IDX_T>&  adj_vector = get<0>(it->second);
              const data::AttrVal&    my_edge_attrs = get<1>(it->second);

              ierr = pack_edge(comm, header_type, key_node, adj_vector, my_edge_attrs,
                               sendpos, sendbuf);
              assert(ierr == 0);
              assert((size_t)sendpos <= sendbuf.size());
            }
        }

      return ierr;
    }

    
    void pack_edge_map (MPI_Comm comm, MPI_Datatype header_type, MPI_Datatype size_type,
                        edge_map_t& prj_edge_map, 
                        size_t &num_packed_edges,
                        int &sendpos,
                        vector<uint8_t> &sendbuf)
    {
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

#ifdef USE_EDGE_DELIM      
      int packsize=0;
      assert(MPI_Pack_size(2, MPI_INT, comm, &packsize) == MPI_SUCCESS);
      sendbuf.resize(sendbuf.size() + packsize, 0);
      assert(MPI_Pack(&rank_edge_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif
      pack_edge_map1 (comm, header_type, size_type, 
                      prj_edge_map, num_packed_edges, sendpos, sendbuf);

#ifdef USE_EDGE_DELIM      
      assert(MPI_Pack(&rank_edge_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                      &sendpos, comm) == MPI_SUCCESS);

#endif
      assert(sendpos <= (int)sendbuf.size());
      
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
     data::AttrVal& edge_attr_values,
     int & recvpos
     )
    {
      
      int ierr;
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      if (edge_attr_num > 0)
        {
#ifdef USE_EDGE_DELIM
          int delim=0;
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm);
          assert(ierr == MPI_SUCCESS);
          if (delim != edge_attr_start_delim)
            {
              printf("rank %d: unpack_edge_attr: recvpos = %d recvbuf_size = %u delim = %d\n", 
                     rank, recvpos, recvbuf_size, delim);
            }
          assert(delim == edge_attr_start_delim);
#endif

          if (!(recvpos < recvbuf_size))
            {
              printf("recvbuf_size = %u recvpos = %d\n", recvbuf_size, recvpos);
            }
          assert(recvpos < recvbuf_size);

          for (size_t k = 0; k < edge_attr_num; k++)
            {
              assert(recvpos < recvbuf_size);
              vector<T> vec;
              vec.resize(num_edges, 0);
              ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                                &vec[0], num_edges, mpi_type, comm);
              assert(ierr == MPI_SUCCESS);
              edge_attr_values.insert(vec);
            }

#ifdef USE_EDGE_DELIM
          delim=0;
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm);
          assert(ierr == MPI_SUCCESS);
          if (delim != edge_attr_end_delim)
            {
              printf("rank %d: unpack_edge_attr: recvpos = %d recvbuf_size = %u delim = %d\n", 
                     rank,  recvpos, recvbuf_size, delim);
            }
          assert(delim == edge_attr_end_delim);
#endif
        }
    }

    int unpack_edge
    (
     MPI_Comm comm,
     MPI_Datatype header_type,
     const vector<uint8_t> &recvbuf,
     const vector<uint32_t> &edge_attr_num,
     NODE_IDX_T &key_node,
     vector<NODE_IDX_T>& adj_vector,
     data::AttrVal& edge_attr_values,
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
      if (delim != edge_start_delim)
        {
          printf("rank %d: unpack_edge: recvpos = %d recvbuf_size = %u delim = %d\n", 
                 rank, recvpos, recvbuf_size, delim);
        }
      assert(delim == edge_start_delim);
#endif
      EdgeHeader header;
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &header, 1, header_type, comm);
      assert(ierr == MPI_SUCCESS);
      key_node = header.key;
      size_t numitems = header.size;
      if (numitems > 0)
        {
          adj_vector.resize(numitems,0);
          ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                            &adj_vector[0], numitems, NODE_IDX_MPI_T, comm);
          assert(ierr == MPI_SUCCESS);
          
          if (!(recvpos <= recvbuf_size))
            {
              printf("rank %d: unpack_edge: recvbuf_size = %u recvpos = %d key_node = %u numitems = %lu\n", 
                     rank, recvbuf_size, recvpos, key_node, numitems);
            }
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<float>(comm, MPI_FLOAT, numitems, edge_attr_num[data::AttrVal::attr_index_float],
                                         recvbuf, recvbuf_size, 
                                         edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint8_t>(comm, MPI_UINT8_T, numitems, edge_attr_num[data::AttrVal::attr_index_uint8],
                                           recvbuf, recvbuf_size, 
                                           edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint16_t>(comm, MPI_UINT16_T, numitems, edge_attr_num[data::AttrVal::attr_index_uint16],
                                            recvbuf, recvbuf_size, 
                                            edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<uint32_t>(comm, MPI_UINT32_T, numitems, edge_attr_num[data::AttrVal::attr_index_uint32],
                                            recvbuf, recvbuf_size, 
                                            edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<int8_t>(comm, MPI_INT8_T, numitems, edge_attr_num[data::AttrVal::attr_index_int8],
                                           recvbuf, recvbuf_size, 
                                           edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<int16_t>(comm, MPI_INT16_T, numitems, edge_attr_num[data::AttrVal::attr_index_int16],
                                            recvbuf, recvbuf_size, 
                                            edge_attr_values, recvpos);
          assert(recvpos <= recvbuf_size);
          unpack_edge_attr_values<int32_t>(comm, MPI_INT32_T, numitems, edge_attr_num[data::AttrVal::attr_index_int32],
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


    void unpack_edge_map (MPI_Comm comm,
                          MPI_Datatype header_type,
                          MPI_Datatype size_type,
                          const vector<uint8_t> &recvbuf,
                          const vector<uint32_t> &edge_attr_num,
                          edge_map_t& prj_edge_map
                          )
    {
      const int recvbuf_size = recvbuf.size();
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      int recvpos = 0;

#ifdef USE_EDGE_DELIM
              int delim=0;
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
              if (delim != rank_edge_start_delim)
                {
                  printf("rank %d: unpack_edge_map: recvpos = %d recvbuf_size = %d delim = %d\n", 
                         rank, recvpos, recvbuf_size, delim);
                }
              assert(delim == rank_edge_start_delim);
#endif
          
       Size sizeval;
       size_t num_recv_items=0; 
              
       // Unpack number of received blocks for this rank
       assert(MPI_Unpack(&recvbuf[0], recvbuf_size, 
                         &recvpos, &sizeval, 1, size_type, comm) ==
              MPI_SUCCESS);
       num_recv_items = sizeval.size;
       if (num_recv_items > 0)
         {
           for (size_t j = 0; j<num_recv_items; j++)
             {
               NODE_IDX_T key_node; 
               vector<NODE_IDX_T> adj_vector;
               data::AttrVal edge_attr_values;
                      
               unpack_edge(comm, header_type, recvbuf, edge_attr_num, 
                           key_node, adj_vector, edge_attr_values, recvpos);
               if (recvpos > (int)recvbuf_size)
                 {
                   printf("rank %d: unpacking projection has reached end of buffer; "
                          "recvpos = %d recvbuf_size = %d j = %lu num_recv_items = %lu\n", 
                          rank, recvpos, recvbuf_size, j, num_recv_items);
                 }
               assert(recvpos <= recvbuf_size);
               
               if (prj_edge_map.find(key_node) == prj_edge_map.end())
                 {
                   prj_edge_map.insert(make_pair(key_node,make_tuple(adj_vector, edge_attr_values)));
                 }
               else
                 {
                   edge_tuple_t et = prj_edge_map[key_node];
                   vector<NODE_IDX_T> &v = get<0>(et);
                   data::AttrVal &a = get<1>(et);
                   v.insert(v.end(),adj_vector.begin(),adj_vector.end());
                   a.append(edge_attr_values);
                   prj_edge_map[key_node] = make_tuple(v,a);
                 }
             }
         }

#ifdef USE_EDGE_DELIM
              delim=0;
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
              if (delim != rank_edge_end_delim)
                {
                  printf("rank %d: unpack_edge_map: recvpos = %d recvbuf_size = %u delim = %d\n", 
                         rank, recvpos, recvbuf_size, delim);
                }
              assert(delim == rank_edge_end_delim);
#endif

    }

    
    
    void pack_rank_edge_map (MPI_Comm comm, MPI_Datatype header_type, MPI_Datatype size_type,
                             rank_edge_map_t& prj_rank_edge_map, 
                             size_t &num_packed_edges,
                             vector<int>& sendcounts,
                             vector<uint8_t> &sendbuf,
                             vector<int> &sdispls)
    {
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      int sendpos = 0;
      vector<int> rank_sequence;
      // Recommended all-to-all communication pattern: start at the current rank, then wrap around;
      // (as opposed to starting at rank 0)
      for (int key_rank = rank; (int)key_rank < size; key_rank++)
        {
          rank_sequence.push_back(key_rank);
        }
      for (int key_rank = 0; (int)key_rank < rank; key_rank++)
        {
          rank_sequence.push_back(key_rank);
        }
      
      for (const int& key_rank : rank_sequence)
        {
          sdispls[key_rank] = sendpos;
          
          auto it1 = prj_rank_edge_map.find(key_rank); 

#ifdef USE_EDGE_DELIM      
          int packsize=0;
          assert(MPI_Pack_size(2, MPI_INT, comm, &packsize) == MPI_SUCCESS);
          sendbuf.resize(sendbuf.size() + packsize, 0);
          assert(MPI_Pack(&rank_edge_start_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                          &sendpos, comm) == MPI_SUCCESS);

#endif
          if (it1 != prj_rank_edge_map.end())
            {
              pack_edge_map1 (comm, header_type, size_type, 
                              it1->second, num_packed_edges, sendpos, sendbuf);
              prj_rank_edge_map.erase(key_rank);
              
            } else
            {
              const edge_map_t empty_edge_map;
              pack_edge_map1 (comm, header_type, size_type, 
                              empty_edge_map, num_packed_edges, sendpos, sendbuf);
            }
          
#ifdef USE_EDGE_DELIM      
          assert(MPI_Pack(&rank_edge_end_delim, 1, MPI_INT, &sendbuf[0], sendbuf.size(),
                          &sendpos, comm) == MPI_SUCCESS);

#endif
          sendbuf.resize(sendbuf.size()+MPI_BSEND_OVERHEAD,0);
          sendpos += MPI_BSEND_OVERHEAD;
          assert(sendpos <= (int)sendbuf.size());
          sendcounts[key_rank] = sendpos - sdispls[key_rank];
        }
      
    }

    

    


    void unpack_rank_edge_map (MPI_Comm comm,
                               MPI_Datatype header_type,
                               MPI_Datatype size_type,
                               const size_t io_size,
                               const vector<uint8_t> &recvbuf,
                               const vector<int>& recvcounts,
                               const vector<int>& rdispls,
                               const vector<uint32_t> &edge_attr_num,
                               edge_map_t& prj_edge_map,
                               uint64_t& num_unpacked_edges
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
              int startpos = recvpos;
              assert(recvpos < recvbuf_size);
#ifdef USE_EDGE_DELIM
              int delim=0;
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
              if (delim != rank_edge_start_delim)
                {
                  printf("rank %d: unpack_rank_edge_map: ridx = %lu recvcounts[%lu] = %d recvpos = %d recvbuf_size = %d delim = %d\n", 
                         rank, ridx, ridx, recvcounts[ridx], recvpos, recvbuf_size, delim);
                  while ((delim != rank_edge_start_delim) && (recvpos < recvbuf_size))
                    {
                      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &delim, 1, MPI_INT, comm) == MPI_SUCCESS);
                      printf("rank %d: recvpos = %d delim = %d\n", rank, recvpos, delim);
                    }
                  
                }
              assert(delim == rank_edge_start_delim);
#endif
              Size sizeval;
              size_t num_recv_items=0; 
              
              // Unpack number of received blocks for this rank
              assert(MPI_Unpack(&recvbuf[0], recvbuf_size, 
                                &recvpos, &sizeval, 1, size_type, comm) ==
                     MPI_SUCCESS);
              num_recv_items = sizeval.size;
              if (num_recv_items > 0)
                {
                  for (size_t j = 0; j<num_recv_items; j++)
                    {
                      NODE_IDX_T key_node; 
                      vector<NODE_IDX_T> adj_vector;
                      data::AttrVal edge_attr_values;
                      
                      unpack_edge(comm, header_type, recvbuf, edge_attr_num, 
                                  key_node, adj_vector, edge_attr_values, recvpos);
                      num_unpacked_edges += adj_vector.size();
                      if (recvpos > (int)recvbuf_size)
                        {
                          printf("rank %d: unpacking projection has reached end of buffer; "
                                 "recvpos = %d recvbuf_size = %d j = %lu num_recv_items = %lu\n", 
                                 rank, recvpos, recvbuf_size, j, num_recv_items);
                        }
                      assert(recvpos <= (int)recvbuf_size);
                      if (prj_edge_map.find(key_node) == prj_edge_map.end())
                        {
                          prj_edge_map.insert(make_pair(key_node,make_tuple(adj_vector, edge_attr_values)));
                        }
                      else
                        {
                          edge_tuple_t et = prj_edge_map[key_node];
                          vector<NODE_IDX_T> &v = get<0>(et);
                          data::AttrVal &a = get<1>(et);
                          v.insert(v.end(),adj_vector.begin(),adj_vector.end());
                          a.append(edge_attr_values);
                          prj_edge_map[key_node] = make_tuple(v,a);
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


  }
}
