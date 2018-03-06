// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file balance_graph_indegree.cc
///
///  Partition graph vertices according to their degree.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection.hh"
#include "cell_populations.hh"
#include "scatter_read_graph.hh"
#include "merge_edge_map.hh"
#include "vertex_degree.hh"
#include "validate_edge_list.hh"

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

    void compute_part_nums
    (
     const size_t&     num_blocks,
     const size_t&     size,
     vector< size_t >& bins
     )
    {
      hsize_t remainder=0, offset=0, buckets=0, num=0;
      bins.resize(size);
      for (size_t i=0; i<size; i++)
        {
          remainder = num_blocks - offset;
          buckets   = (size - i);
          num       = remainder / buckets;
          bins[i]   = num;
          offset    += num;
        }
    }

  
    /*****************************************************************************
     * Main balancing routine
     *****************************************************************************/

    int balance_graph_indegree
    (
     MPI_Comm comm,
     const std::string& input_file_name,
     const std::vector< std::pair<std::string, std::string> > prj_names,
     const size_t io_size,
     const size_t Nparts,
     std::vector<NODE_IDX_T> &parts,
     std::vector<double> &part_weights
     )
    {
      int status=0;
    
      int rank, size;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

    
      // Read population info to determine total_num_nodes
      size_t local_num_nodes, total_num_nodes, local_num_edges, total_num_edges;

      vector <string> edge_attr_name_spaces;
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(cell::read_population_ranges(comm, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      map<NODE_IDX_T, rank_t> node_rank_map;
      compute_node_rank_map(size, total_num_nodes, node_rank_map);
    
      // read the edges
      vector < edge_map_t > prj_vector;
      vector < map <string, vector <vector<string> > > > edge_attr_name_vector;
      scatter_read_graph (comm,
                          EdgeMapDst,
                          input_file_name,
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
      
      vector < std::map<NODE_IDX_T, size_t> > vertex_indegree_maps;
      vertex_degree (comm, prj_vector, vertex_indegree_maps);

      std::vector<double> vertex_norm_indegrees(total_num_nodes, 0.0);

      for (const map< NODE_IDX_T, size_t >& vertex_indegree_map : vertex_indegree_maps)
        {
          uint64_t sum_indegree=0, nz_indegree=0;
          
          for (auto it = vertex_indegree_map.begin(); it != vertex_indegree_map.end(); it++)
            {
              size_t degree = it->second;
              sum_indegree = sum_indegree + degree;
              if (degree > 0) nz_indegree++;
            }
          for (auto it = vertex_indegree_map.begin(); it != vertex_indegree_map.end(); it++)
            {
              double norm_indegree = (double)it->second / (double)sum_indegree;
              vertex_norm_indegrees[it->first] += norm_indegree;
            }
          
        }


      vector<NODE_IDX_T> node_idx_vector(total_num_nodes, 0);
      for (NODE_IDX_T i=0; i<total_num_nodes; i++)
        {
          node_idx_vector[i] = i;
        }

      // Sort node indices according to in-degree
      std::sort(node_idx_vector.begin(), node_idx_vector.end(),
                    [&] (NODE_IDX_T const& a, NODE_IDX_T const& b)
                    {
                  return vertex_norm_indegrees[a] > vertex_norm_indegrees[b];
                });
      part_weights.resize(Nparts);
      std::vector< size_t > part_nums;
      compute_part_nums(total_num_nodes,Nparts,part_nums);
      std::map<NODE_IDX_T, size_t> parts_map;
      NODE_IDX_T vidx=0;
      for (size_t p=0; p<Nparts; p++)
        {
          double part_norm_indegree=0.0;
          while (vidx < total_num_nodes)
            {
              NODE_IDX_T n = node_idx_vector[vidx];
              if (vertex_norm_indegrees[n] > 0)
                {
                  parts_map.insert(make_pair(n, p));
                  part_norm_indegree = part_norm_indegree + vertex_norm_indegrees[n];
                  part_nums[p]--;
                }
              vidx++;
            }
          part_weights[p] = part_norm_indegree;
        }


      vector<size_t> part_idx_vector;
      part_idx_vector.resize(Nparts);
      for (size_t i=0; i<Nparts; i++)
        {
          part_idx_vector[i] = i;
        }
      // Sort partition indices according to number of nodes assigned to them
      std::sort(part_idx_vector.begin(), part_idx_vector.end(),
                [&] (size_t const& a, size_t const& b)
                {
                  return part_nums[a] < part_nums[b];
                });
      
      vidx=0;
      while (vidx < total_num_nodes)
        {
          for (size_t pidx=0; pidx<Nparts; pidx++)
            {
              size_t p = part_idx_vector[pidx];
              if (part_nums[p] > 0)
                {
                  NODE_IDX_T n = node_idx_vector[vidx];
                  if (parts_map.find(n) == parts_map.end())
                    {
                      parts_map.insert(make_pair(n, p));
                      part_nums[p]--;
                    }
                  vidx++;
                }
            }
        }
      
      parts.resize(total_num_nodes);
      for (NODE_IDX_T n=0; n<total_num_nodes; n++)
        {
          parts[n] = parts_map[n];
        }
      
      return status;
    }
  
  }
}
