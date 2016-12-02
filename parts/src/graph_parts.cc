// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file reader.cc
///
///  Graph partitioning functions.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#include "debug.hh"
#include "ngh5paths.hh"
#include "ngh5types.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "graph_reader.hh"

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

namespace ngh5
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

  void merge_edge_map (const vector < edge_map_t > &prj_vector,
                       map<NODE_IDX_T, vector<NODE_IDX_T> > &edge_map)
  {
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        edge_map_t prj_edge_map = prj_vector[i];
        if (prj_edge_map.size() > 0)
          {
            for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
              {
                NODE_IDX_T dst   = it->first;
                edge_tuple_t& et = it->second;
                
                const vector<NODE_IDX_T> src_vector = get<0>(et);
                
                if (edge_map.find(dst) == edge_map.end())
                  {
                    edge_map.insert(make_pair(dst,src_vector));
                  }
                else
                  {
                    vector<NODE_IDX_T> &v = edge_map[dst];
                    v.insert(v.end(),src_vector.begin(),src_vector.end());
                    edge_map[dst] = v;
                  }
              }
          }
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
    assert(read_population_ranges(comm, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

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

    printf("parts: rank %d: finished scatter_graph\n", rank);
    // Combine the edges from all projections into a single edge map
    map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
    merge_edge_map (prj_vector, edge_map);
    printf("parts: rank %d: finished merging map\n", rank);
    prj_vector.clear();
    
    // Needed by parmetis
    vector<idx_t> vtxdist;
    vector<idx_t> xadj;
    vector<idx_t> adjncy;
    idx_t *vwgt=NULL, *adjwgt=NULL;
    idx_t wgtflag = 0; // indicates if the graph is weighted (0 = no weights)
    idx_t numflag = 0; // indicates array numbering scheme (0: C-style; 1: Fortran-style)
    idx_t ncon    = 1; // number of weights per vertex
    idx_t nparts  = Nparts;
    vector <real_t> tpwgts;
    real_t ubvec = 1.05; 
    idx_t options[4], edgecut;
    
    // Common for every rank:
    // determines which graph nodes are assigned to which MPI rank
    compute_vtxdist(size, total_num_nodes, vtxdist);

    // Specific to each rank:
    //
    // the adjacency list of vertex i is stored in array adjncy
    // starting at index xadj[i] and ending at (but not including)
    // index xadj[i + 1]
    size_t adjncy_offset = 0;
    for (idx_t i = vtxdist[rank]; i<vtxdist[rank+1]; i++)
      {
        auto it = edge_map.find(i);
        if (it != edge_map.end())
          {
            NODE_IDX_T dst = it->first;
            const vector<NODE_IDX_T> src_vector = it->second;
            
            xadj.push_back(adjncy_offset);
            adjncy.insert(adjncy.end(),src_vector.begin(),src_vector.end());
        
            adjncy_offset = adjncy_offset + src_vector.size();
          }
        else
          {
            xadj.push_back(adjncy_offset);
          }
      }
    xadj.push_back(adjncy.size());

    edge_map.clear();

    tpwgts.resize(Nparts); // fraction of vertex weight that should be distributed to each partition
    for (size_t i = 0; i < Nparts; i++)
      {
        tpwgts[i] = 1.0/Nparts;
      }
    
    parts.resize (vtxdist[rank+1]-vtxdist[rank]); // resize to number of locally stored vertices
    status = ParMETIS_V3_PartKway (&vtxdist[0],&xadj[0],&adjncy[0],
                                   vwgt,adjwgt,&wgtflag,&numflag,&ncon,&nparts,
                                   &tpwgts[0],&ubvec,options,&edgecut,&parts[0],
                                   &comm);
    if (status != METIS_OK)
      {
        throw_err("ParMETIS error");
      }
    
    return status;
  }
  
}