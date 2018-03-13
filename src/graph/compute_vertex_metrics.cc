// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file compute_vertex_metrics.cc
///
///  Computes vertex metrics in the graph,
///
///  Copyright (C) 2016-2018 Project NeuroH5.
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
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < num_nodes; i++)
        {
            node_rank_map.insert(make_pair(i, i%num_ranks));
        }
    }


    void append_vertex_degree_map (MPI_Comm comm, const map<NODE_IDX_T, rank_t>& node_rank_map,
                                   const std::vector< std::pair<std::string, std::string> >& prj_names,
                                   size_t total_num_nodes,
                                   const std::vector < map< NODE_IDX_T, size_t> > & vertex_degree_maps,
                                   const string& label, const string& norm_label,
                                   const string& output_file_name)
    {
      int srank, ssize;
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);

      size_t rank, size;
      rank = (size_t)srank;
      size = (size_t)ssize;

      size_t prj_index=0;
      for (const map< NODE_IDX_T, size_t>& vertex_degree_map : vertex_degree_maps)
        {
          uint64_t sum_degree=0, nz_degree=0;
          
          for (auto it = vertex_degree_map.begin(); it != vertex_degree_map.end(); it++)
            {
              size_t degree = it->second;
              sum_degree = sum_degree + degree;
              if (degree > 0) nz_degree++;
            }
          
          std::vector<float> vertex_norm_degrees;
          vertex_norm_degrees.resize(total_num_nodes);
          for (auto it = vertex_degree_map.begin(); it != vertex_degree_map.end(); it++)
            {
              float norm_degree = (float)(it->second) / (float)sum_degree;
              vertex_norm_degrees[it->first] += norm_degree;
            }
          
          vector <NODE_IDX_T> node_id;
          vector <ATTR_PTR_T> attr_ptr;
          vector <size_t> vertex_degree_value;
          vector <float> vertex_norm_degree_value;
          
          attr_ptr.push_back(0);
          for (auto it=node_rank_map.begin(); it != node_rank_map.end(); it++)
            {
              if (it->second == rank)
                {
                  const auto it_degree_value = vertex_degree_map.find(it->first);
                  if (it_degree_value != vertex_degree_map.cend())
                    {
                      node_id.push_back(it->first);
                      attr_ptr.push_back(attr_ptr.back() + 1);
                      vertex_degree_value.push_back(it_degree_value->second);
                      vertex_norm_degree_value.push_back(vertex_norm_degrees[it->first]);
                    }
                }
            }

          string src_pop_name = prj_names[prj_index].first;
          string dst_pop_name = prj_names[prj_index].second;
          
          graph::append_node_attribute (comm, output_file_name, "Vertex Metrics",
                                        label + " " + src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_degree_value);
          graph::append_node_attribute (comm, output_file_name, "Vertex Metrics",
                                        norm_label + " " + src_pop_name + " -> " + dst_pop_name,
                                        node_id, attr_ptr, vertex_norm_degree_value);

          prj_index++;
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
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);

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

      {
        vector < std::map< NODE_IDX_T, size_t> > vertex_indegree_maps; 
        vertex_degree (prj_vector, false, vertex_indegree_maps);
        append_vertex_degree_map (comm, node_rank_map, prj_names,
                                  total_num_nodes, vertex_indegree_maps,
                                  "Indegree", "Norm indegree",
                                  file_name);
      }

      {
        vector < std::map< NODE_IDX_T, size_t> > vertex_unique_indegree_maps;
        vertex_degree (prj_vector, true, vertex_unique_indegree_maps);
        append_vertex_degree_map (comm, node_rank_map, prj_names,
                                  total_num_nodes, vertex_unique_indegree_maps,
                                  "Unique indegree", "Norm unique indegree",
                                  file_name);
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
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);

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
      
      {
        vector < std::map< NODE_IDX_T, size_t> > vertex_outdegree_maps; 
        vertex_degree (prj_vector, false, vertex_outdegree_maps);
        append_vertex_degree_map (comm, node_rank_map, prj_names,
                                  total_num_nodes, vertex_outdegree_maps,
                                  "Outdegree", "Norm outdegree",
                                  file_name);
      }

      {
        vector < std::map< NODE_IDX_T, size_t> > vertex_unique_outdegree_maps;
        vertex_degree (prj_vector, true, vertex_unique_outdegree_maps);
        append_vertex_degree_map (comm, node_rank_map, prj_names,
                                  total_num_nodes, vertex_unique_outdegree_maps,
                                  "Unique outdegree", "Norm unique outdegree",
                                  file_name);
      }
      
      return status;
    }


  }
}
