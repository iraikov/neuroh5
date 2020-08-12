// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph_info.cc
///
///  Top-level functions for reading graph metadata.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "edge_attributes.hh"

#include "read_projection_info.hh"
#include "cell_populations.hh"
#include "read_graph_info.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    int read_graph_info
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const vector<string>& edge_attr_name_spaces,
     const bool            read_node_index, 
     vector< pair<string, string> >& prj_names,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     std::vector<std::vector<NODE_IDX_T>>& prj_node_index
     )
    {
      int status = 0;
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
      
      // read the population info
      size_t total_num_nodes;
      vector<pop_range_t> pop_vector;
      vector< pair<pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      throw_assert_nomsg(cell::read_population_combos(comm, file_name, pop_pairs) >= 0);
      throw_assert_nomsg(cell::read_population_ranges
             (comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);
      throw_assert_nomsg(cell::read_population_labels(comm, file_name, pop_labels) >= 0);

      // read the edges
      for (auto const& it : pop_pairs)
        {
          size_t local_prj_num_nodes;

          pop_t src_pop_idx = it.first, dst_pop_idx = it.second;
          string src_pop_name, dst_pop_name;
          bool src_pop_set = false, dst_pop_set = false;
          
          for (auto const& label_it : pop_labels)
            {
              if (it.first == src_pop_idx)
                {
                  src_pop_name = it.second;
                  src_pop_set = true;
                }
              if (it.first == dst_pop_idx)
                {
                  dst_pop_name = it.second;
                  dst_pop_set = true;
                }
            }

          throw_assert (src_pop_set && dst_pop_set,
                        "read_graph_info: invalid source or destination population index");
          
          NODE_IDX_T dst_start = pop_vector[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_vector[src_pop_idx].start;

          throw_assert_nomsg(graph::read_projection_info
                             (comm, file_name, edge_attr_name_spaces, read_node_index,
                              pop_ranges, pop_pairs,
                              src_pop_name, dst_pop_name, 
                              src_start, dst_start, 
                              prj_names, edge_attr_names_vector,
                              prj_node_index) >= 0);
        }

      return 0;
    }




  }
}
