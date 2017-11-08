// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_degree.cc
///
///  Calculate vertex (in/out)degree from an edge map.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include <map>
#include <vector>
#include <mpi.h>

#undef NDEBUG
#include <cassert>

using namespace neuroh5;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    int vertex_degree (MPI_Comm comm,
                       const size_t global_num_nodes,
                       const vector < edge_map_t >& prj_vector,
                       vector < map< NODE_IDX_T, uint32_t > > &degree_maps)
    {
      int status=0; 
      int ssize;
      assert(MPI_Comm_size(comm, &ssize) >= 0);
      size_t size;
      size = (size_t)ssize;

      for (const map<NODE_IDX_T, edge_tuple_t >& edge_map : prj_vector)
        {
          uint32_t local_num_nodes=0;
          vector <uint32_t> local_degree_vector;
          vector <NODE_IDX_T> local_node_id_vector;
          vector <uint32_t> degree_vector;
          vector <NODE_IDX_T> node_id_vector;
          local_num_nodes = edge_map.size();
          
          for (auto it = edge_map.begin(); it != edge_map.end(); ++it)
            {
              local_node_id_vector.push_back(it->first);
              const vector<NODE_IDX_T>& adj_vector = get<0>(it->second);
              uint32_t degree = adj_vector.size();
              local_degree_vector.push_back(degree);
            }
      
          vector<uint32_t> num_nodes_vector(size, 0);
          num_nodes_vector.resize(size,0);
          assert(MPI_Allgather(&local_num_nodes, 1, MPI_UINT32_T, 
                               &num_nodes_vector[0], 1, MPI_UINT32_T, comm) >= 0);
          vector<int> recvcounts, rdispls;
          recvcounts.resize(size,0); 
          rdispls.resize(size,0); 
          
          for (size_t i = 0; i<size; i++)
            {
              recvcounts[i] = num_nodes_vector[i];
            }
          for (size_t i = 1; i<size; ++i)
            {
              rdispls[i] = rdispls[i-1] + recvcounts[i];
            }
          
          degree_vector.resize(global_num_nodes);
          node_id_vector.resize(global_num_nodes);
          assert(MPI_Allgatherv(&local_degree_vector[0], 
                                local_num_nodes, MPI_UINT32_T,
                                &degree_vector[0], &recvcounts[0], &rdispls[0], MPI_UINT32_T,
                                comm) >= 0);
          assert(MPI_Allgatherv(&local_node_id_vector[0], 
                                local_num_nodes, MPI_NODE_IDX_T,
                                &node_id_vector[0], &recvcounts[0], &rdispls[0], MPI_NODE_IDX_T,
                                comm) >= 0);

          map< NODE_IDX_T, uint32_t > degree_map;
          for (size_t i = 0; i < node_id_vector.size(); i++)
            {
              degree_map.insert(make_pair(node_id_vector[i], degree_vector[i]));
            }

          degree_maps.push_back(degree_map);
        }

      return status;
    }
  }
  
}
