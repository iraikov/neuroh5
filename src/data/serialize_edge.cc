// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file serialize_edge.cc
///
///  Top-level functions for serializing/deserializing graphs edges.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "serialize_edge.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/archives/binary.hpp"

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace data
  {
    
    void serialize_rank_edge_map (const size_t num_ranks,
                                  const size_t start_rank,
                                  const rank_edge_map_t& prj_rank_edge_map, 
                                  size_t &num_packed_edges,
                                  vector<int>& sendcounts,
                                  vector<uint8_t> &sendbuf,
                                  vector<int> &sdispls)
    {
      vector<int> rank_sequence;
      // Recommended all-to-all communication pattern: start at the current rank, then wrap around;
      // (as opposed to starting at rank 0)
      for (int key_rank = start_rank; (int)key_rank < num_ranks; key_rank++)
        {
          rank_sequence.push_back(key_rank);
        }
      for (int key_rank = 0; (int)key_rank < start_rank; key_rank++)
        {
          rank_sequence.push_back(key_rank);
        }

      size_t sendpos = 0;
      for (const int& key_rank : rank_sequence)
        {
          sdispls[key_rank] = sendpos;
          
          auto it1 = prj_rank_edge_map.find(key_rank); 
          std::stringstream ss;
          const edge_map_t& edge_map = it1->second;
          
          for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
            {
              NODE_IDX_T key_node = it->first;
              const vector<NODE_IDX_T>&  adj_vector = get<0>(it->second);

              num_packed_edges += adj_vector.size();
            }
          
          {
            cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
            oarchive(edge_map); // Write the data to the archive

          } // archive goes out of scope, ensuring all contents are flushed

          sendpos = sendbuf.size();
          copy(istream_iterator<char>(ss), istream_iterator<char>(), back_inserter(sendbuf));
          sendcounts[key_rank] = sendpos - sdispls[key_rank];
        }
      
    }


    void deserialize_rank_edge_map (const size_t num_ranks,
                                    const vector<uint8_t> &recvbuf,
                                    const vector<int>& recvcounts,
                                    const vector<int>& rdispls,
                                    const vector<uint32_t> &edge_attr_num,
                                    edge_map_t& prj_edge_map,
                                    uint64_t& num_unpacked_edges
                                    )
    {
      const int recvbuf_size = recvbuf.size();

      for (size_t ridx = 0; (int)ridx < num_ranks; ridx++)
        {
          if (recvcounts[ridx] > 0)
            {
              int recvsize  = recvcounts[ridx];
              int recvpos   = rdispls[ridx];
              int startpos  = recvpos;
              assert(recvpos < recvbuf_size);
              edge_map_t edge_map;

              {
                string s = string(recvbuf.begin()+startpos, recvbuf.begin()+startpos+recvsize);
                stringstream ss(s);

                cereal::BinaryInputArchive iarchive(ss); // Create an input archive
                
                iarchive(edge_map); // Read the data from the archive
              }
              
              for (auto it = edge_map.cbegin(); it != edge_map.cend(); ++it)
                {
                  NODE_IDX_T key_node = it->first;
                  const vector<NODE_IDX_T>&  adj_vector = get<0>(it->second);
                  const data::AttrVal&    edge_attr_values = get<1>(it->second);
                      
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
        }
    }

  }
}
