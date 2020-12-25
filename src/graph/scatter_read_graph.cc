// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "scatter_read_projection.hh"
#include "scatter_read_graph.hh"
#include "throw_assert.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    


    int scatter_read_graph
    (
     MPI_Comm                      all_comm,
     const EdgeMapType             edge_map_type,
     const std::string&            file_name,
     const int                     io_size,
     const vector<string> &        attr_namespaces,
     const vector< pair<string, string> >&         prj_names,
     // A vector that maps nodes to compute ranks
     const node_rank_map_t&  node_rank_map,
     vector < edge_map_t >& prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t                       &local_num_nodes,
     size_t                       &total_num_nodes,
     size_t                       &local_num_edges,
     size_t                       &total_num_edges
     )
    {
      int ierr = 0;
      // The set of compute ranks for which the current I/O rank is responsible
      set< pair<pop_t, pop_t> > pop_pairs;
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      vector< pair<pop_t, string> > pop_labels;
      
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);
          
       throw_assert_nomsg(cell::read_population_ranges
                          (all_comm, file_name, pop_ranges, pop_vector, total_num_nodes)
                          >= 0);
       throw_assert_nomsg(cell::read_population_labels(all_comm, file_name, pop_labels) >= 0);
       throw_assert_nomsg(cell::read_population_combos(all_comm, file_name, pop_pairs)  >= 0);
          
      // For each projection, I/O ranks read the edges and scatter
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          hsize_t total_read_blocks;

          string src_pop_name = prj_names[i].first;
          string dst_pop_name = prj_names[i].second;

          scatter_read_projection(all_comm, io_size, edge_map_type,
                                  file_name, src_pop_name, dst_pop_name, attr_namespaces,
                                  node_rank_map, pop_vector, pop_ranges, pop_labels, pop_pairs,
                                  prj_vector, edge_attr_names_vector, 
                                  local_num_nodes, local_num_edges, total_num_edges,
                                  total_read_blocks);

        }
      return ierr;
    }

  }
  
}
