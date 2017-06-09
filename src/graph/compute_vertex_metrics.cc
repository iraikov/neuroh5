// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file compute_vertex_metrics.cc
///
///  Computes vertex metrics in the graph,
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include "read_projection.hh"
#include "cell_populations.hh"
#include "scatter_graph.hh"
#include "merge_edge_map.hh"
#include "vertex_degree.hh"
#include "validate_edge_list.hh"
#include "node_attributes.hh"

#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>

#include <mpi.h>

using namespace std;

namespace neuroh5
{
  namespace graph
  {

    void throw_err(char const* err_message)
    {
      fprintf(stderr, "Error: %s\n", err_message);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    void throw_err(char const* err_message, int32_t task)
    {
      fprintf(stderr, "Task %d Error: %s\n", task, err_message);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    void throw_err(char const* err_message, int32_t task, int32_t thread)
    {
      fprintf(stderr, "Task %d Thread %d Error: %s\n", task, thread, err_message);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign each node to a rank 
    void compute_node_rank_map
    (
     size_t num_ranks,
     size_t num_nodes,
     map< NODE_IDX_T, rank_t > &node_rank_map
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
    
      for (size_t i=0; i<num_ranks; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_ranks - i;
          for (size_t j = 0; j < remainder / buckets; j++)
            {
              node_rank_map.insert(make_pair(offset+j, i));
            }
          offset    += remainder / buckets;
        }
    }

  
    int compute_vertex_indegree
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::vector< std::pair<std::string, std::string> > prj_names,
     const size_t io_size
     )
    {
      int status=0;
    
      int srank, ssize;
      assert(MPI_Comm_size(comm, &ssize) >= 0);
      assert(MPI_Comm_rank(comm, &srank) >= 0);

      size_t rank, size;
      rank = (size_t)srank;
      size = (size_t)ssize;
    
      // Read population info to determine total_num_nodes
      size_t total_num_nodes,
        local_num_edges, total_num_edges;

      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(cell::read_population_ranges(comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      map<NODE_IDX_T, rank_t> node_rank_map;
      compute_node_rank_map(size, total_num_nodes, node_rank_map);
    
      // read the edges
      vector < vector <vector<string>> > edge_attr_name_vector;
      vector < edge_map_t > prj_vector;
      scatter_graph (comm,
                     EdgeMapDst,
                     file_name,
                     io_size,
                     false,
                     prj_names,
                     node_rank_map,
                     prj_vector,
                     edge_attr_name_vector,
                     total_num_nodes,
                     local_num_edges,
                     total_num_edges);

      // Combine the edges from all projections into a single edge map
      map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
      merge_edge_map (prj_vector, edge_map);
      prj_vector.clear();

      uint64_t sum_indegree=0, nz_indegree=0;
      std::vector<uint32_t> vertex_indegrees;
      vertex_degree (comm, total_num_nodes, edge_map, vertex_indegrees);
      edge_map.clear();

      for(size_t v=0; v<total_num_nodes; v++)
        {
          uint32_t degree = vertex_indegrees[v];
          sum_indegree = sum_indegree + degree;
          if (degree > 0) nz_indegree++;
        }
      std::vector<float> vertex_norm_indegrees;
      vertex_norm_indegrees.resize(total_num_nodes);
      for(size_t v=0; v<total_num_nodes; v++)
        {
          float norm_indegree = (float)vertex_indegrees[v] / (float)sum_indegree;
          vertex_norm_indegrees[v] = norm_indegree;
        }
      
      vector <NODE_IDX_T> node_id;
      vector <ATTR_PTR_T> attr_ptr;
      vector <uint32_t> vertex_indegree_value;
      vector <float> vertex_norm_indegree_value;

      attr_ptr.push_back(0);
      for (auto it=node_rank_map.begin(); it != node_rank_map.end(); it++)
        {
          if (it->second == rank)
            {
              node_id.push_back(it->first);
              attr_ptr.push_back(attr_ptr.back() + 1);
              vertex_indegree_value.push_back(vertex_indegrees[it->first]);
              vertex_norm_indegree_value.push_back(vertex_norm_indegrees[it->first]);
            }
        }
      
      graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex indegree",
                                   node_id, attr_ptr, vertex_indegree_value);
      graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex norm indegree",
                                    node_id, attr_ptr, vertex_norm_indegree_value);

      return status;
    }

    
    int compute_vertex_outdegree
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::vector< std::pair<std::string, std::string> > prj_names,
     const size_t io_size
     )
    {
      int status=0;
    
      int srank, ssize;
      assert(MPI_Comm_size(comm, &ssize) >= 0);
      assert(MPI_Comm_rank(comm, &srank) >= 0);

      size_t rank, size;
      rank = (size_t)srank;
      size = (size_t)ssize;
    
      // Read population info to determine total_num_nodes
      size_t total_num_nodes,
        local_num_edges, total_num_edges;

      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(cell::read_population_ranges(comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      map<NODE_IDX_T, rank_t> node_rank_map;
      compute_node_rank_map(size, total_num_nodes, node_rank_map);
    
      // read the edges
      vector < vector <vector<string>> > edge_attr_name_vector;
      vector < edge_map_t > prj_vector;
      scatter_graph (comm,
                     EdgeMapSrc,
                     file_name,
                     io_size,
                     false,
                     prj_names,
                     node_rank_map,
                     prj_vector,
                     edge_attr_name_vector,
                     total_num_nodes,
                     local_num_edges,
                     total_num_edges);
      
      // Combine the edges from all projections into a single edge map
      map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
      merge_edge_map (prj_vector, edge_map);

      prj_vector.clear();

      uint64_t sum_outdegree=0, nz_outdegree=0;
      std::vector<uint32_t> vertex_outdegrees;
      vertex_degree (comm, total_num_nodes, edge_map, vertex_outdegrees);
      edge_map.clear();

      for(size_t v=0; v<total_num_nodes; v++)
        {
          uint32_t degree = vertex_outdegrees[v];
          sum_outdegree = sum_outdegree + degree;
          if (degree > 0) nz_outdegree++;
        }
      std::vector<float> vertex_norm_outdegrees;
      vertex_norm_outdegrees.resize(total_num_nodes);
      for(size_t v=0; v<total_num_nodes; v++)
        {
          float norm_outdegree = (float)vertex_outdegrees[v] / (float)sum_outdegree;
          vertex_norm_outdegrees[v] = norm_outdegree;
        }
      
      
      vector <ATTR_PTR_T> attr_ptr;
      vector <NODE_IDX_T> node_id;
      vector <uint32_t> vertex_outdegree_value;
      vector <float> vertex_norm_outdegree_value;

      attr_ptr.push_back(0);
      for (auto it=node_rank_map.begin(); it != node_rank_map.end(); it++)
        {
          if (it->second == rank)
            {
              node_id.push_back(it->first);
              attr_ptr.push_back(attr_ptr.back() + 1);
              vertex_outdegree_value.push_back(vertex_outdegrees[it->first]);
              vertex_norm_outdegree_value.push_back(vertex_norm_outdegrees[it->first]);
            }
        }

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      
      graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex outdegree",
                                    node_id, attr_ptr, vertex_outdegree_value);
      graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex norm outdegree",
                                    node_id, attr_ptr, vertex_norm_outdegree_value);

      return status;
    }


  }
}
