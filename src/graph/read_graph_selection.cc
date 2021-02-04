// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph_selection.cc
///
///  Top-level functions for reading specified subsets of graphs in
///  DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include "edge_attributes.hh"
#include "cell_populations.hh"
#include "read_projection_selection.hh"
#include "read_graph_selection.hh"
#include "throw_assert.hh"
#include "debug.hh"
#include "mpi_debug.hh"

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

      prj_vector.clear();
      
      size_t selection_size = selection.size();
      int data_color = 2;
      
      MPI_Comm data_comm;
      // In cases where some ranks do not have any data to read, split
      // the communicator, so that collective operations can be executed
      // only on the ranks that do have data.
      if (selection_size > 0)
        {
          MPI_Comm_split(comm,data_color,0,&data_comm);
        }
      else
        {
          MPI_Comm_split(comm,0,0,&data_comm);
        }
      MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

      if (selection_size > 0)
        {
          int rank=-1, size=-1;

          throw_assert(MPI_Comm_size(data_comm, &size) == MPI_SUCCESS, "unable to obtain MPI communicator size");
          throw_assert(MPI_Comm_rank(data_comm, &rank) == MPI_SUCCESS, "unable to obtain MPI communicator rank");
          
          // read the population info
          pop_label_map_t pop_labels;
          pop_range_map_t pop_ranges;
          set< pair<pop_t, pop_t> > pop_pairs;
          throw_assert(cell::read_population_combos(data_comm, file_name, pop_pairs) >= 0,
                       "unable to read valid projection combination");
          throw_assert(cell::read_population_ranges
                       (data_comm, file_name, pop_ranges, total_num_nodes) >= 0,
                       "unable to read population ranges");
          throw_assert(cell::read_population_labels(data_comm, file_name, pop_labels) >= 0,
                       "unable to read population labels");
          pop_search_range_map_t pop_search_ranges;
          for (auto &x : pop_ranges)
            {
              pop_search_ranges.insert(make_pair(x.second.start, make_pair(x.second.count, x.first)));
            }

          // read the edges
          for (size_t i = 0; i < prj_names.size(); i++)
            {
              size_t local_prj_num_nodes=0;
              size_t local_prj_num_edges=0;
              size_t total_prj_num_edges=0;
              
              //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());
              
              string src_pop_name = prj_names[i].first, dst_pop_name = prj_names[i].second;

              pop_t dst_pop_idx=0, src_pop_idx=0;
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
              NODE_IDX_T dst_end = pop_ranges[dst_pop_idx].start + pop_ranges[dst_pop_idx].count;
              NODE_IDX_T src_start = pop_ranges[src_pop_idx].start;

              mpi::MPI_DEBUG(data_comm, "read_graph_selection: src_pop_name = ", src_pop_name,
                             " dst_pop_name = ", dst_pop_name,
                             " dst_start = ", dst_start,
                             " src_start = ", src_start);

              bool selection_found = false;
              for (auto gid : selection)
                {
                  if ((dst_start <= gid) && (gid < dst_end))
                    {
                      selection_found = true;
                      break;
                    }
                }

              if (selection_found)
                {
                  
                  throw_assert(graph::read_projection_selection
                               (data_comm, file_name, pop_search_ranges, pop_pairs,
                                src_pop_name, dst_pop_name, 
                                src_start, dst_start, edge_attr_name_spaces, 
                                selection, prj_vector, edge_attr_names_vector,
                                local_prj_num_nodes,
                                local_prj_num_edges, total_prj_num_edges) >= 0,
                               "error in read_projection_selection");
                  
                  total_num_edges = total_num_edges + total_prj_num_edges;
                  local_num_edges = local_num_edges + local_prj_num_edges;
                }
              else
                {
                  edge_map_t prj_edge_map;
                  map <string, vector < vector<string> > > edge_attr_names;
                  prj_vector.push_back(prj_edge_map);
                  edge_attr_names_vector.push_back(edge_attr_names);
                }
              
            }

          size_t sum_local_num_edges = 0;
          status = MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
                              MPI_SIZE_T, MPI_SUM, 0, data_comm);
          throw_assert(status == MPI_SUCCESS,
                       "error in MPI_Reduce");
        }

      throw_assert(MPI_Barrier(data_comm) == MPI_SUCCESS,
                   "error in MPI_Barrier");
      throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                   "error in MPI_Comm_free");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "error in MPI_Barrier");
#endif

      return 0;
    }




  }
}
