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
#include <metis.h>
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
                NODE_IDX_T src   = it->first;
                edge_tuple_t& et = it->second;
                
                const vector<NODE_IDX_T> dst_vector = get<0>(et);
                
                if (edge_map.find(dst) == edge_map.end())
                  {
                    edge_map.insert(make_pair(src,dst_vector));
                  }
                else
                  {
                    vector<NODE_IDX_T> &v = edge_map[dst];
                    v.insert(v.end(),dst_vector.begin(),dst_vector.end());
                    edge_map[src] = v;
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

    printf("parts: rank %d: io_size = %lu Nparts = %lu\n", rank, io_size, Nparts);
    
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
    scatter_graph_src (comm,
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
    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 6\n", rank);

    // Combine the edges from all projections into a single edge map
    map<NODE_IDX_T, vector<NODE_IDX_T> > edge_map;
    merge_edge_map (prj_vector, edge_map);
    printf("parts: rank %d: finished merging map\n", rank);
    prj_vector.clear();

    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 5\n", rank);
    
    // Needed by parmetis
    vector<idx_t> vtxdist;
    idx_t *vwgt=NULL, *adjwgt=NULL;
    idx_t wgtflag = 0; // indicates if the graph is weighted (0 = no weights)
    idx_t numflag = 0; // indicates array numbering scheme (0: C-style; 1: Fortran-style)
    idx_t ncon    = 1; // number of weights per vertex
    idx_t nparts  = Nparts;

    printf("parts: rank %d: nparts = %u\n", rank, nparts);
    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 4\n", rank);

    // fraction of vertex weight that should be distributed to each partition
    vector <real_t> tpwgts(nparts, 1.0/(real_t)nparts);
    real_t ubvec = 1.05; 
    idx_t options[4], edgecut;

    options[0] = 1; // use user-supplied options 
    options[1] = PARMETIS_DBGLVL_TIME | PARMETIS_DBGLVL_INFO | PARMETIS_DBGLVL_PROGRESS; // debug level
    options[2] = 0;

    printf("parts: rank %d: tpwgts.size() = %lu\n", rank, tpwgts.size());

    real_t sum = 0;
    for (size_t i = 0; i<tpwgts.size(); i++)
      {
        sum = sum + tpwgts[i];
      }
    printf("parts: rank %d: sum = %g\n", rank, sum);

    // Common for every rank:
    // determines which graph nodes are assigned to which MPI rank
    compute_vtxdist(size, total_num_nodes, vtxdist);
    printf("parts: rank %d: building parmetis structure\n", rank);
    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 3\n", rank);

    // Specific to each rank:
    //
    // the adjacency list of vertex i is stored in array adjncy
    // starting at index xadj[i] and ending at (but not including)
    // index xadj[i + 1]
    size_t adjncy_offset = 0;
    vector<idx_t> xadj, adjncy;
    printf("parts: rank %d: vtxdist[rank] = %u vtxdist[rank+1] = %u edge_map.size() = %lu\n", 
           rank, vtxdist[rank], vtxdist[rank+1], edge_map.size());
    for (NODE_IDX_T i = vtxdist[rank]; i<vtxdist[rank+1]; i++)
      {
        auto it = edge_map.find(i);
        if (it == edge_map.end())
          {
            xadj.push_back(adjncy_offset);
          }
        else
          {
            NODE_IDX_T src = it->first;
            const vector<NODE_IDX_T> &dst_vector = it->second;
            printf("parts: rank %d: dst = %u\n", rank, dst);
            
            xadj.push_back(adjncy_offset);
            for (size_t j = 0; j<dst_vector.size(); j++)
              {
                adjncy.push_back(dst_vector[j]);
              }
            adjncy_offset = adjncy_offset + dst_vector.size();
          }
      }
    printf("parts: rank %d: adjncy.size() = %lu\n", rank, adjncy.size());
    //assert(adjncy.size() > 0);
    xadj.push_back(adjncy.size());
    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 2\n", rank);
    edge_map.clear();
    printf("parts: rank %d: cleared edge_map\n", rank);
    size_t num_local_vtxs = vtxdist[rank+1]-vtxdist[rank];
    printf("parts: rank %d: num_local_vtxs = %lu\n", rank, num_local_vtxs);
    assert(num_local_vtxs > 0);
    parts.resize (num_local_vtxs); // resize to number of locally stored vertices
    printf("parts: rank %d: resized parts\n", rank);
    MPI_Barrier(comm);
    printf("parts: rank %d: after barrier 1\n", rank);
    status = ParMETIS_V3_PartKway (vtxdist.data(),xadj.data(),adjncy.data(),
                                   vwgt,adjwgt,&wgtflag,&numflag,&ncon,&nparts,
                                   tpwgts.data(),&ubvec,options,&edgecut,parts.data(),
                                   &comm);
    if (status != METIS_OK)
      {
        throw_err("ParMETIS error");
      }
    
    return status;
  }
  
}
