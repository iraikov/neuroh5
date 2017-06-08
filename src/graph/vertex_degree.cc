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
                       const map<NODE_IDX_T, vector<NODE_IDX_T> > &edge_map,
                       vector< uint32_t > &degree_vector)
    {
      int status=0; uint32_t local_num_nodes;
      vector <uint32_t> local_degree_vector;
      local_num_nodes = edge_map.size();
      
      for (auto it = edge_map.begin(); it != edge_map.end(); ++it)
        {
          const vector<NODE_IDX_T>& adj_vector = it->second;
          uint32_t degree = adj_vector.size();

          local_degree_vector.push_back(degree);
        }
      
      int srank, ssize;
      assert(MPI_Comm_size(comm, &ssize) >= 0);
      assert(MPI_Comm_rank(comm, &srank) >= 0);
      size_t rank, size;
      rank = (size_t)srank;
      size = (size_t)ssize;
      
      vector<uint32_t> num_nodes_vector;
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
      assert(MPI_Allgatherv(&local_degree_vector[0], 
                           local_num_nodes, MPI_UINT32_T,
                           &degree_vector[0], &recvcounts[0], &rdispls[0], MPI_UINT32_T,
                           comm) >= 0);

      return status;
    }
  }
  
}
