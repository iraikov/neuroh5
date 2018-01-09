// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file compute_vertex_metrics.cc
///
///  Computes vertex metrics in the graph,
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection.hh"
#include "cell_populations.hh"
#include "scatter_read_graph.hh"
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
using namespace neuroh5;

namespace neuroh5
{
  namespace graph
  {
    // Assign each node to a rank 
    static void compute_node_rank_map
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
      size_t local_num_nodes, total_num_nodes, local_num_edges, total_num_edges;

      vector<string> edge_attr_name_spaces;
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(cell::read_population_ranges(comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      map<NODE_IDX_T, rank_t> node_rank_map;
      compute_node_rank_map(size, total_num_nodes, node_rank_map);
    
      // read the edges
      vector < map <string, vector <vector<string> > > > edge_attr_name_vector;
      vector < edge_map_t > prj_vector;
      scatter_read_graph (comm,
                          EdgeMapDst,
                          file_name,
                          io_size,
                          edge_attr_name_spaces,
                          prj_names,
                          node_rank_map,
                          prj_vector,
                          edge_attr_name_vector,
                          local_num_nodes,
                          total_num_nodes,
                          local_num_edges,
                          total_num_edges);

      vector < std::map< NODE_IDX_T, uint32_t> > vertex_indegree_maps;
      vertex_degree (comm, total_num_nodes, prj_vector, vertex_indegree_maps);

      size_t prj_index=0;
      for (const map< NODE_IDX_T, uint32_t>& vertex_indegree_map : vertex_indegree_maps)
        {
          uint64_t sum_indegree=0, nz_indegree=0;
          
          for (auto it = vertex_indegree_map.begin(); it != vertex_indegree_map.end(); it++)
            {
              uint32_t degree = it->second;
              sum_indegree = sum_indegree + degree;
              if (degree > 0) nz_indegree++;
            }
          
          std::vector<float> vertex_norm_indegrees;
          vertex_norm_indegrees.resize(total_num_nodes);
          for (auto it = vertex_indegree_map.begin(); it != vertex_indegree_map.end(); it++)
            {
              float norm_indegree = (float)(it->second) / (float)sum_indegree;
              vertex_norm_indegrees[it->first] += norm_indegree;
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
                  const auto it_indegree_value = vertex_indegree_map.find(it->first);
                  if (it_indegree_value != vertex_indegree_map.cend())
                    {
                      vertex_indegree_value.push_back(it_indegree_value->second);
                    }
                  else
                    {
                      vertex_indegree_value.push_back(0);
                    }
                  vertex_norm_indegree_value.push_back(vertex_norm_indegrees[it->first]);
                }
            }

          string src_pop_name = prj_names[prj_index].first;
          string dst_pop_name = prj_names[prj_index].second;
          
          graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Indegree " +
                                        src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_indegree_value);
          graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Norm indegree " +
                                        src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_norm_indegree_value);

          prj_index++;
        }

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
      size_t local_num_nodes, total_num_nodes,
        local_num_edges, total_num_edges;

      vector<string> edge_attr_name_spaces;
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(cell::read_population_ranges(comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      map<NODE_IDX_T, rank_t> node_rank_map;
      compute_node_rank_map(size, total_num_nodes, node_rank_map);
    
      // read the edges
      vector < map <string, vector <vector<string> > > > edge_attr_name_vector;
      vector < edge_map_t > prj_vector;
      scatter_read_graph (comm,
                          EdgeMapSrc,
                          file_name,
                          io_size,
                          edge_attr_name_spaces,
                          prj_names,
                          node_rank_map,
                          prj_vector,
                          edge_attr_name_vector,
                          local_num_nodes,
                          total_num_nodes,
                          local_num_edges,
                          total_num_edges);
      
      // Combine the edges from all projections into a single edge map
      map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;

      uint64_t sum_outdegree=0, nz_outdegree=0;
      vector < std::map<NODE_IDX_T, uint32_t> > vertex_outdegree_maps;
      vertex_degree (comm, total_num_nodes, prj_vector, vertex_outdegree_maps);

      prj_vector.clear();

      size_t prj_index=0;
      for (const map< NODE_IDX_T, uint32_t>& vertex_outdegree_map : vertex_outdegree_maps)
        {

          for (auto it = vertex_outdegree_map.begin(); it != vertex_outdegree_map.end(); it++)
            {
              uint32_t degree = it->second;
              sum_outdegree = sum_outdegree + degree;
              if (degree > 0) nz_outdegree++;
            }
          std::vector<float> vertex_norm_outdegrees;
          vertex_norm_outdegrees.resize(total_num_nodes);
          for (auto it = vertex_outdegree_map.begin(); it != vertex_outdegree_map.end(); it++)
            {
              float norm_outdegree = (float)(it->second) / (float)sum_outdegree;
              vertex_norm_outdegrees[it->first] = norm_outdegree;
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

                  const auto it_outdegree_value = vertex_outdegree_map.find(it->first);
                  if (it_outdegree_value != vertex_outdegree_map.cend())
                    {
                      vertex_outdegree_value.push_back(it_outdegree_value->second);
                    }
                  else
                    {
                      vertex_outdegree_value.push_back(0);
                    }
                  vertex_norm_outdegree_value.push_back(vertex_norm_outdegrees[it->first]);
                }
            }
          string src_pop_name = prj_names[prj_index].first;
          string dst_pop_name = prj_names[prj_index].second;

          graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex outdegree " +
                                        src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_outdegree_value);
          graph::append_node_attribute (comm, file_name, "Vertex Metrics", "Vertex norm outdegree " +
                                        src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_norm_outdegree_value);

          prj_index++;
        }
      
      return status;
    }


  }
}
