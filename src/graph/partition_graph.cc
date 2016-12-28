// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_parts.cc
///
///  Graph partitioning functions.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "graph_reader.hh"
#include "merge_edge_map.hh"
#include "vertex_degree.hh"
#include "read_population.hh"
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

#include <mpi.h>
#include <parmetis.h>

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


    // Calculate the starting and end graph node index for each rank
    void compute_vtxdist
    (
     size_t num_ranks,
     size_t num_nodes,
     vector< idx_t > &vtxdist
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
    
      vtxdist.resize(num_ranks+1);
      for (size_t i=0; i<num_ranks; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_ranks - i;
          vtxdist[i] = offset;
          offset    += remainder / buckets;
        }
      vtxdist[num_ranks] = num_nodes;
    }

    // Calculate the starting and end graph node index for each partition
    void compute_partdist
    (
     size_t num_parts,
     size_t num_nodes,
     vector< NODE_IDX_T > &partdist
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
    
      partdist.resize(num_parts+1);
      for (size_t i=0; i<num_parts; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_parts - i;
          partdist[i] = offset;
          offset    += remainder / buckets;
        }
      partdist[num_parts] = num_nodes;
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

  
    /*****************************************************************************
     * Main partitioning routine
     *****************************************************************************/

    int partition_graph
    (
     MPI_Comm comm,
     const std::string& input_file_name,
     const std::vector<std::string> prj_names,
     const size_t io_size,
     const size_t Nparts,
     std::vector<idx_t> &parts
     )
    {
      int status;
    
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

    
      // Read population info to determine total_num_nodes
      size_t local_num_nodes, total_num_nodes,
        local_num_edges, total_num_edges;

      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(io::hdf5::read_population_ranges(comm, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

      // A vector that maps nodes to compute ranks
      vector<rank_t> node_rank_vector;
      compute_node_rank_vector(size, total_num_nodes, node_rank_vector);
    
      // read the edges
      vector < edge_map_t > prj_vector;
      scatter_graph (comm,
                     input_file_name,
                     io_size,
                     false,
                     prj_names,
                     node_rank_vector,
                     prj_vector,
                     total_num_nodes,
                     local_num_edges,
                     total_num_edges);
      
      DEBUG("rank ", rank, ": parts: after scatter");
      // Combine the edges from all projections into a single edge map
      map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
      merge_edge_map (prj_vector, edge_map);

      DEBUG("rank ", rank, ": parts: after merge");

      prj_vector.clear();

      uint64_t sum_indegree=0;
      std::vector<uint32_t> vertex_indegrees;
      std::vector<float> vertex_indegree_fractions;
      vertex_degree (comm, total_num_nodes, edge_map, vertex_indegrees);
      for(size_t v; v<total_num_nodes; v++)
        {
          uint32_t degree = vertex_indegrees[v];
          sum_indegree = sum_indegree + degree;
        }
      
      // normalize vertex indegrees by sum_indegree
      vertex_indegree_fractions.resize(total_num_nodes,0.0);
      for(size_t v; v<total_num_nodes; v++)
        {
          uint32_t degree = vertex_indegrees[v];
          float fraction = (float)degree / (float)sum_indegree;
          vertex_indegree_fractions[v] = fraction;
        }
      // global_max_indegree, global_min_indegree
      DEBUG("rank ", rank, ": parts: after vertex_degree: sum_indegree = ", sum_indegree);
      
      // Needed by parmetis
      vector<idx_t> vtxdist;
      vector<idx_t> xadj;
      vector<idx_t> adjncy;
      vector<idx_t> vwgts;
      idx_t *adjwgts=NULL;
      idx_t wgtflag = 2; // indicates if the graph is weighted (2 = vertex weights only)
      idx_t numflag = 0; // indicates array numbering scheme (0: C-style; 1: Fortran-style)
      idx_t ncon    = 2; // number of weights per vertex; the second
                         // weight is calculated from the in-degree of
                         // each vertex
      idx_t nparts  = Nparts;
      vector <real_t> tpwgts;
      real_t ubvec[2] = {1.5, 1.5}; 
      idx_t options[4], edgecut;
      options[0] = 1;
      options[1] = 3;

      // Common for every rank:
      // determines which graph nodes are assigned to which MPI rank
      compute_vtxdist(size, total_num_nodes, vtxdist);
      DEBUG("rank ", rank, ": parts: after compute_vtxdist");

      // Specific to each rank:
      //
      // the adjacency list of vertex i is stored in array adjncy
      // starting at index xadj[i] and ending at (but not including)
      // index xadj[i + 1]
      idx_t adjncy_offset = 0;
      for (idx_t i = vtxdist[rank]; i<vtxdist[rank+1]; i++)
        {
          xadj.push_back(adjncy_offset);
          auto it = edge_map.find(i);
          if (it != edge_map.end())
            {
              NODE_IDX_T dst = it->first;
              const vector<NODE_IDX_T> src_vector = it->second;
            
              adjncy.insert(adjncy.end(),src_vector.begin(),src_vector.end());
              adjncy_offset = adjncy_offset + src_vector.size();
            }
          vwgts.push_back(vertex_indegrees[i]);
        }
      xadj.push_back(adjncy_offset);
      DEBUG("rank ", rank, ": parts: after xadj");
      edge_map.clear();

      tpwgts.resize(ncon*Nparts,0.0); // fraction of vertex weight that should be distributed to each partition

      vector<NODE_IDX_T> partdist;
      // determines which graph nodes are assigned to which partition
      compute_partdist(Nparts, total_num_nodes, partdist);
      for (size_t i = 0, p = 0; i < ncon*Nparts; i+=ncon, p++)
        {
          tpwgts[i+0] = 1.0/(float)Nparts;
          NODE_IDX_T start = partdist[p];
          NODE_IDX_T end   = partdist[p+1];
          for (NODE_IDX_T j=start; j<end; j++)
            {
              tpwgts[i+1] += vertex_indegree_fractions[j];
            }
        }
    
      DEBUG("rank ", rank, ": parts: before parts.resize");
      parts.resize (vtxdist[rank+1]-vtxdist[rank]); // resize to number of locally stored vertices
      DEBUG("rank ", rank, ": calling parmetis");
      status = ParMETIS_V3_PartKway (&vtxdist[0],&xadj[0],&adjncy[0],
                                     &vwgts[0],adjwgts,&wgtflag,&numflag,&ncon,&nparts,
                                     &tpwgts[0],ubvec,options,&edgecut,&parts[0],
                                     &comm);
      if (status != METIS_OK)
        {
          throw_err("ParMETIS error");
        }
    
      return status;
    }
  
  }
}
