// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file compute_vertex_metrics.cc
///
///  Computes vertex metrics in the graph,
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "read_dbs_projection.hh"
#include "population_reader.hh"
#include "scatter_graph.hh"
#include "merge_edge_map.hh"
#include "vertex_degree.hh"
#include "read_population.hh"
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
using namespace ngh5::model;

namespace ngh5
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
    void compute_node_rank_vector
    (
     size_t num_ranks,
     size_t num_nodes,
     vector< rank_t > &node_rank_vector
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
    
      node_rank_vector.resize(num_nodes);
      for (size_t i=0; i<num_ranks; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_ranks - i;
          for (size_t j = 0; j < remainder / buckets; j++)
            {
              node_rank_vector[offset+j] = i;
            }
          offset    += remainder / buckets;
        }
    }

  
    int compute_vertex_metrics
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::vector<std::string> prj_names,
     const size_t io_size
     )
    {
      int status=0;
    
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

    
      // Read population info to determine total_num_nodes
      size_t local_num_nodes, total_num_nodes,
        local_num_edges, total_num_edges;

      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(io::hdf5::read_population_ranges(comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      vector<rank_t> node_rank_vector;
      compute_node_rank_vector(size, total_num_nodes, node_rank_vector);
    
      // read the edges
      vector < vector <vector<string>> > edge_attr_name_vector;
      vector < edge_map_t > prj_vector;
      scatter_graph (comm,
                     EdgeMapDst,
                     file_name,
                     io_size,
                     false,
                     prj_names,
                     node_rank_vector,
                     prj_vector,
                     edge_attr_name_vector,
                     total_num_nodes,
                     local_num_edges,
                     total_num_edges);
      
      DEBUG("rank ", rank, ": parts: after scatter");
      // Combine the edges from all projections into a single edge map
      map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
      merge_edge_map (prj_vector, edge_map);

      DEBUG("rank ", rank, ": parts: after merge");

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
      
      float mean_norm_indegree = 1.0 / (float)nz_indegree;

      vector <NODE_IDX_T> node_id;
      vector <uint32_t> vertex_indegree_value;
      vector <float> vertex_norm_indegree_value;

      for (NODE_IDX_T i=0; i<node_rank_vector.size(); i++)
        {
          if (node_rank_vector[i] == rank)
            {
              node_id.push_back(i);
              vertex_indegree_value.push_back(vertex_indegrees[i]);
              vertex_norm_indegree_value.push_back(vertex_norm_indegrees[i]);
            }
        }

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      assert(file >= 0);
      
      ngh5::io::hdf5::write_node_attribute (file, "Vertex indegree", node_id, vertex_indegree_value);
      ngh5::io::hdf5::write_node_attribute (file, "Vertex norm indegree", node_id, vertex_norm_indegree_value);

      assert(H5Fclose(file) >= 0);
      assert(H5Pclose(fapl) >= 0);

      return status;
    }
  
  }
}
