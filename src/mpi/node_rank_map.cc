// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file node_rank_map.cc
///
///  Function for creating a mapping of graph nodes to MPI ranks.
///
///  Copyright (C) 2025 Project NeuroH5.
//==============================================================================

#include <mpi.h>

#include <vector>
#include <map>
#include <algorithm>
#include <cassert>

#include "mpi_debug.hh"
#include "throw_assert.hh"
#include "neuroh5_types.hh"
#include "node_rank_map.hh"

using namespace std;


namespace neuroh5
{

  namespace mpi
  {


    void compute_node_rank_map
    (
     MPI_Comm comm,
     set<rank_t> &rank_set,
     vector< NODE_IDX_T > &local_node_index,
     size_t &total_num_nodes,
     map<NODE_IDX_T, rank_t> &node_rank_map
     )
    {
      int rank, size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      
      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };

      
      // Step 1: Gather all nodes from all ranks to create a global list
      MPI_Request request;
      size_t local_num_nodes = local_node_index.size();
      vector<size_t> sendbuf_num_nodes(size, num_nodes);
      vector<size_t> recvbuf_num_nodes(size);
      vector<int> recvcounts(size, 0);
      vector<int> displs(size+1, 0);
      
      total_num_nodes=0;
      
      throw_assert(MPI_Iallgather(&sendbuf_num_nodes[0], 1, MPI_SIZE_T, 
                                  &recvbuf_num_nodes[0], 1, MPI_SIZE_T, 
                                  comm, &request) == MPI_SUCCESS,
                   "compute_node_rank_map: error in MPI_Iallgather");
      
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                     "compute_node_rank_map: error in MPI_Wait");
      
      // Calculate displacements for gathering
      displs[0] = 0;
      for (size_t p = 0; p < size; p++)
        {
          total_num_nodes = total_num_nodes + recvbuf_num_nodes[p];
          displs[p+1] = displs[p] + recvbuf_num_nodes[p];
          recvcounts[p] = recvbuf_num_nodes[p];
        }

      vector <NODE_IDX_T> node_index(total_num_nodes, 0);

      throw_assert(MPI_Iallgatherv(&local_node_index[0], local_num_nodes, MPI_NODE_IDX_T,
                                   &node_index[0], &recvcounts[0], &displs[0], MPI_NODE_IDX_T,
                                   comm,
                                   &request) == MPI_SUCCESS,
                   "compute_node_rank_map: error in MPI_Iallgatherv");
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "compute_node_rank_map: error in MPI_Wait");

      vector<size_t> p = sort_permutation(node_index, compare_nodes);
      apply_permutation_in_place(node_index, p);
      
      // Verify we have the expected number of nodes
      throw_assert(node_index.size() == total_num_nodes,
                   "compute_node_rank_map: mismatch in number of nodes");                   
      
      // Clear the mapping
      node_rank_map.clear();

      // Assign nodes to ranks in round-robin manner
      rank_t r=0; 
      for (size_t i = 0; i < node_index.size(); i++)
        {
          while (io_rank_set.count(r) == 0)
            {
              r++;
              if ((unsigned int)size <= r) r=0;
            }
          node_rank_map.insert(make_pair(node_index[i], r++));
        }
    }
    
  }
}
