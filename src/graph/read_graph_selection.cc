// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph.cc
///
///  Top-level functions for reading specified subsets of graphs in
///  DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================

#include "debug.hh"
#include "mpi_debug.hh"
#include "edge_attributes.hh"
#include "cell_populations.hh"
#include "read_projection_selection.hh"
#include "read_graph_selection.hh"
#include "throw_assert.hh"

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    int read_graph_selection
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const vector<string>& edge_attr_name_spaces,
     const vector< pair<string, string> >& prj_names,
     const std::vector<NODE_IDX_T>&  selection,
     std::vector<edge_map_t>& prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&              total_num_nodes,
     size_t&              local_num_edges,
     size_t&              total_num_edges
     )
    {
      int status = 0;
      int rank, size;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "unable to obtain MPI communicator rank");
      
      // read the population info
      vector<pop_range_t> pop_vector;
      vector< pair<pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      throw_assert(cell::read_population_combos(comm, file_name, pop_pairs) >= 0,
                   "unable to read valid projection combination");
      throw_assert(cell::read_population_ranges
                   (comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0,
                   "unable to read population ranges");
      throw_assert(cell::read_population_labels(comm, file_name, pop_labels) >= 0,
                   "unable to read population labels");

      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          size_t local_prj_num_nodes;
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
          throw_assert(dst_pop_set && src_pop_set,
                       "unable to determine destination or source population");
      
          NODE_IDX_T dst_start = pop_vector[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_vector[src_pop_idx].start;

          mpi::MPI_DEBUG(comm, "read_graph: src_pop_name = ", src_pop_name,
                         " dst_pop_name = ", dst_pop_name,
                         " dst_start = ", dst_start,
                         " src_start = ", src_start);

          throw_assert(graph::read_projection_selection
                       (comm, file_name, pop_ranges, pop_pairs,
                        src_pop_name, dst_pop_name, 
                        src_start, dst_start, edge_attr_name_spaces, 
                        selection, prj_vector, edge_attr_names_vector,
                        local_prj_num_nodes,
                        local_prj_num_edges, total_prj_num_edges) >= 0,
                       "error in read_projection_selection");

          mpi::MPI_DEBUG(comm, "read_graph: projection ", i, " has a total of ", total_prj_num_edges, " edges");
          
          total_num_edges = total_num_edges + total_prj_num_edges;
          local_num_edges = local_num_edges + local_prj_num_edges;
        }

      size_t sum_local_num_edges = 0;
      status = MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
                          MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
      throw_assert(status == MPI_SUCCESS,
                   "error in MPI_Reduce");

      return 0;
    }




  }
}
