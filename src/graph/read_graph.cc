// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "edge_attributes.hh"

#include "read_projection.hh"
#include "cell_populations.hh"
#include "read_graph.hh"

#undef NDEBUG
#include <cassert>

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    int read_graph
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const vector<string>& edge_attr_name_spaces,
     const vector< pair<string, string> >& prj_names,
     vector<prj_tuple_t>& prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&              total_num_nodes,
     size_t&              local_num_edges,
     size_t&              total_num_edges
     )
    {
      int status = 0;
      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      
      // read the population info
      vector<pop_range_t> pop_vector;
      vector< pair<pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      assert(cell::read_population_combos(comm, file_name, pop_pairs) >= 0);
      assert(cell::read_population_ranges
             (comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);
      assert(cell::read_population_labels(comm, file_name, pop_labels) >= 0);

      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          size_t local_prj_num_edges;
          size_t total_prj_num_edges;

          //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

          string src_pop_name = prj_names[i].first, dst_pop_name = prj_names[i].second;
          uint32_t dst_pop_idx = 0, src_pop_idx = 0;
          bool src_pop_set = false, dst_pop_set = false;
      
          for (size_t i=0; i< pop_labels.size(); i++)
            {
              if (src_pop_name == get<1>(pop_labels[i]))
                {
                  src_pop_idx = get<0>(pop_labels[i]);
                  src_pop_set = true;
                }
              if (dst_pop_name == get<1>(pop_labels[i]))
                {
                  dst_pop_idx = get<0>(pop_labels[i]);
                  dst_pop_set = true;
                }
            }
          assert(dst_pop_set && src_pop_set);
      
          DEBUG("reader: after reading destination and source population");

          NODE_IDX_T dst_start = pop_vector[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_vector[src_pop_idx].start;

          DEBUG(" src_pop_name = ", src_pop_name,
                " dst_pop_name = ", dst_pop_name,
                " dst_start = ", dst_start,
                " src_start = ", src_start,
                "\n");

          assert(graph::read_projection
                 (comm, file_name, pop_ranges, pop_pairs,
                  src_pop_name, dst_pop_name, 
                  dst_start, src_start, edge_attr_name_spaces, 
                  prj_vector, edge_attr_names_vector,
                  local_prj_num_edges, total_prj_num_edges) >= 0);

          DEBUG("reader: projection ", i, " has a total of ", total_prj_num_edges, " edges");

          total_num_edges = total_num_edges + total_prj_num_edges;
          local_num_edges = local_num_edges + local_prj_num_edges;
        }

      size_t sum_local_num_edges = 0;
      status = MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
                          MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
      assert(status == MPI_SUCCESS);
      
      if (rank == 0)
        {
          if (sum_local_num_edges != total_num_edges)
            {
              printf("sum_local_num_edges = %lu total_num_edges = %lu\n",
                     sum_local_num_edges, total_num_edges);
            }
          assert(sum_local_num_edges == total_num_edges);
        }

      return 0;
    }


    
    int read_graph_serial
    (
     const std::string&   file_name,
     const vector<string>& attr_namespaces,
     const vector< pair<string, string> >& prj_names,
     const vector<string>& edge_attr_name_spaces,
     vector<prj_tuple_t>& prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&              total_num_nodes,
     size_t&              total_num_edges
     )
    {
      // read the population info
      vector<pop_range_t> pop_vector;
      vector< pair<pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      
      assert(cell::read_population_combos_serial
             (file_name, pop_pairs) >= 0);
      assert(cell::read_population_ranges_serial
             (file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);
      assert(cell::read_population_labels_serial
             (file_name, pop_labels) >= 0);

      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          NODE_IDX_T dst_start, src_start;
          NamedAttrVal edge_attr_values;
          size_t total_prj_num_edges;

          //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

          string src_pop_name = prj_names[i].first, dst_pop_name = prj_names[i].second;
          uint32_t dst_pop_idx = 0, src_pop_idx = 0;
          bool src_pop_set = false, dst_pop_set = false;
      
          for (size_t i=0; i< pop_labels.size(); i++)
            {
              if (src_pop_name == get<1>(pop_labels[i]))
                {
                  src_pop_idx = get<0>(pop_labels[i]);
                  src_pop_set = true;
                }
              if (dst_pop_name == get<1>(pop_labels[i]))
                {
                  dst_pop_idx = get<0>(pop_labels[i]);
                  dst_pop_set = true;
                }
            }
          assert(dst_pop_set && src_pop_set);
      
          DEBUG("reader: after reading destination and source population");

          dst_start = pop_vector[dst_pop_idx].start;
          src_start = pop_vector[src_pop_idx].start;

          DEBUG(" dst_start = ", dst_start,
                " src_start = ", src_start,
                "\n");

          assert(graph::read_projection_serial
                 (file_name, pop_ranges, pop_pairs,
                  src_pop_name, dst_pop_name, dst_start, src_start,
                  edge_attr_name_spaces, prj_vector, edge_attr_names_vector,
                  total_prj_num_edges) >= 0);

          DEBUG("reader: projection ", i, " has a total of ", total_prj_num_edges, " edges");

          total_num_edges = total_num_edges + total_prj_num_edges;
        }

      return 0;
    }


  }
}
