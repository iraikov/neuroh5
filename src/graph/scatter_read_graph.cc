// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================


#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "scatter_read_projection.hh"
#include "scatter_read_graph.hh"
#include "throw_assert.hh"
#include "debug.hh"

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
      pop_range_map_t pop_ranges;
      pop_label_map_t pop_labels;
      
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);
          
       throw_assert_nomsg(cell::read_population_ranges
                          (all_comm, file_name, pop_ranges, total_num_nodes)
                          >= 0);
       throw_assert_nomsg(cell::read_population_labels(all_comm, file_name, pop_labels) >= 0);
       throw_assert_nomsg(cell::read_population_combos(all_comm, file_name, pop_pairs)  >= 0);

       pop_search_range_map_t pop_search_ranges;
       for (auto &x : pop_ranges)
         {
           pop_search_ranges.insert(make_pair(x.second.start, make_pair(x.second.count, x.first)));
         }
          
      // For each projection, I/O ranks read the edges and scatter
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          hsize_t total_read_blocks;

          string src_pop_name = prj_names[i].first;
          string dst_pop_name = prj_names[i].second;

          pop_t dst_pop_idx = 0, src_pop_idx = 0;
          bool src_pop_set = false, dst_pop_set = false;
      
          for (auto &x : pop_labels)
            {
              if (src_pop_name == get<1>(x))
                {
                  src_pop_idx = get<0>(x);
                  src_pop_set = true;
                }
              if (dst_pop_name == get<1>(x))
                {
                  dst_pop_idx = get<0>(x);
                  dst_pop_set = true;
                }
            }
          throw_assert_nomsg(dst_pop_set && src_pop_set);

          NODE_IDX_T dst_start = pop_ranges[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_ranges[src_pop_idx].start;

          scatter_read_projection(all_comm, io_size, edge_map_type,
                                  file_name, src_pop_name, dst_pop_name, 
                                  src_start, dst_start,
                                  attr_namespaces,
                                  node_rank_map, pop_search_ranges, pop_pairs,
                                  prj_vector, edge_attr_names_vector, 
                                  local_num_nodes, local_num_edges, total_num_edges,
                                  total_read_blocks);
#ifdef NEUROH5_DEBUG
          MPI_Barrier(all_comm); 
#endif
        }
      return ierr;
    }

  }
  
}
