// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file bcast_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection_datasets.hh"
#include "edge_attributes.hh"
#include "cell_populations.hh"
#include "validate_edge_list.hh"
#include "bcast_template.hh"
#include "append_edge_map.hh"
#include "serialize_edge.hh"
#include "serialize_data.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace graph
  {
    
    /*****************************************************************************
     * Load and broadcast edge data structures 
     *****************************************************************************/

    int bcast_projection (MPI_Comm all_comm, MPI_Comm io_comm,
                          const EdgeMapType edge_map_type,
                          const string& file_name,
                          const string& src_pop_name, 
                          const string& dst_pop_name, 
                          const vector< string >& attr_namespaces,
                          const vector<pop_range_t>& pop_vector,
                          const map<NODE_IDX_T,pair<uint32_t,pop_t> >& pop_ranges,
                          const set< pair<pop_t, pop_t> >& pop_pairs,
                          vector < edge_map_t >& prj_vector,
                          vector < vector < vector<string> > > & edge_attr_names_vector)
                          
    {

      int rank, size;
      assert(MPI_Comm_size(all_comm, &size) >= 0);
      assert(MPI_Comm_rank(all_comm, &rank) >= 0);

      vector<char> sendbuf; 
      vector<NODE_IDX_T> send_edges, recv_edges, total_recv_edges;
      edge_map_t prj_edge_map;
      vector<size_t> edge_attr_num;
      vector<vector<string>> edge_attr_names;
      vector< pair<pop_t, string> > pop_labels;
      size_t num_edges = 0, total_prj_num_edges = 0;
      
      DEBUG("projection ", src_pop_name, " -> ", dst_pop_name, "\n");
      assert(cell::read_population_labels(all_comm, file_name, pop_labels) >= 0);


      if (rank == 0)
        {
          DST_BLK_PTR_T block_base;
          DST_PTR_T edge_base, edge_count;
          NODE_IDX_T dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;
          vector< pair<string,hid_t> > edge_attr_info;
          data::NamedAttrVal edge_attr_values;
          
          uint32_t dst_pop_idx=0, src_pop_idx=0;
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

          dst_start = pop_vector[dst_pop_idx].start;
          src_start = pop_vector[src_pop_idx].start;

          DEBUG("bcast: reading projection ", src_pop_name, " -> ", dst_pop_name);
          assert(hdf5::read_projection_datasets(io_comm, file_name, src_pop_name, dst_pop_name,
                                                dst_start, src_start, block_base, edge_base,
                                                dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                total_prj_num_edges) >= 0);
          
          DEBUG("bcast: validating projection ", src_pop_name, " -> ", dst_pop_name);
          // validate the edges
          assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                    pop_ranges, pop_pairs) == true);
          
          
          edge_count = src_idx.size();

          for (string& attr_namespace : attr_namespaces) 
            {
              vector< pair<string,hid_t> > edge_attr_info;
              assert(graph::get_edge_attributes(file_name, src_pop_name, dst_pop_name,
                                                attr_namespace, edge_attr_info) >= 0);
              assert(graph::num_edge_attributes(edge_attr_info, edge_attr_num) >= 0);
              assert(graph::read_all_edge_attributes(io_comm, file_name, src_pop_name, dst_pop_name,
                                                     attr_namespace, edge_base, edge_count,
                                                     edge_attr_info, edge_attr_values) >= 0);
            }

          // append to the edge map
          
          assert(data::append_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                       edge_attr_values, num_edges, prj_edge_map,
                                       edge_map_type) >= 0);
          edge_attr_values.attr_names(edge_attr_names);

          
          // ensure that all edges in the projection have been read and appended to edge_list
          assert(num_edges == src_idx.size());
          
          size_t num_packed_edges = 0; 
          DEBUG("bcast: packing edge data from projection ", src_pop_name, " -> ", dst_pop_name);
          data::serialize_edge_map (prj_edge_map, num_packed_edges, sendbuf);

          // ensure the correct number of edges is being packed
          assert(num_packed_edges == num_edges);
          DEBUG("bcast: finished packing edge data from projection ", src_pop_name, " -> ", dst_pop_name);

        } // rank == 0
    
      // 0. Broadcast the number of attributes of each type to all ranks
      edge_attr_num.resize(data::AttrVal::num_attr_types, 0);
      assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_SIZE_T, 0, all_comm) == MPI_SUCCESS);
      {
        vector<char> names_sendbuf; uint32_t names_sendbuf_size=0;
        if (rank == 0)
          {
            data::serialize_data(edge_attr_names, names_sendbuf);
            names_sendbuf_size = sendbuf.size();
          }

        assert(MPI_Bcast(&names_sendbuf_size, 1, MPI_UINT32_T, 0, all_comm) >= 0);
        names_sendbuf.resize(names_sendbuf_size);
        assert(MPI_Bcast(&names_sendbuf[0], names_sendbuf_size, MPI_CHAR, 0, all_comm) >= 0);
        
        if (rank != 0)
          {
            data::deserialize_data(names_sendbuf, edge_attr_names);
          }
      }
      
      uint32_t sendbuf_size = sendbuf.size();
      assert(MPI_Bcast(&sendbuf_size, 1, MPI_UINT32_T, 0, all_comm) == MPI_SUCCESS);
      sendbuf.resize(sendbuf_size);
      assert(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_PACKED, 0, all_comm) == MPI_SUCCESS);
          
      size_t num_unpacked_edges = 0, num_unpacked_nodes = 0; 
      if (rank > 0)
        {
          data::deserialize_edge_map (sendbuf, edge_attr_num, prj_edge_map,
                                      num_unpacked_nodes, num_unpacked_edges);
      
          DEBUG("bcast: finished unpacking edges for projection ", src_pop_name, " -> ", dst_pop_name);
        }
      
      prj_vector.push_back(prj_edge_map);
      edge_attr_names_vector.push_back(edge_attr_names);
      assert(MPI_Barrier(all_comm) == MPI_SUCCESS);

      return 0;
    }


    int bcast_graph
    (
     MPI_Comm                      all_comm,
     const EdgeMapType             edge_map_type,
     const std::string&            file_name,
     const vector< string >&       attr_namespaces,
     const vector< pair<string,string> >& prj_names,
     vector < edge_map_t >& prj_vector,
     vector < vector <vector<string>> >& edge_attr_names_vector,
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
      size_t prj_size = 0;
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

  
      int rank, size;
      assert(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);
      
      
      // Am I an I/O rank?
      if (rank == 0)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
      
          // read the population info
          assert(cell::read_population_combos(io_comm, file_name, pop_pairs)
                 >= 0);
          assert(cell::read_population_ranges
                 (io_comm, file_name, pop_ranges, pop_vector, total_num_nodes)
                 >= 0);
          prj_size = prj_names.size();
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);

      assert(MPI_Bcast(&prj_size, 1, MPI_SIZE_T, 0, all_comm) == MPI_SUCCESS);
      DEBUG("rank ", rank, ": bcast: after bcast: prj_size = ", prj_size);

      // For each projection, I/O ranks read the edges and scatter
      for (size_t i = 0; i < prj_size; i++)
        {
          bcast_projection(all_comm, io_comm, size_type, file_name,
                           prj_names[i].first, prj_names[i].second,
                           attr_namespaces, pop_vector, pop_ranges, pop_pairs,
                           prj_vector, edge_attr_names_vector);
                             
        }
      MPI_Comm_free(&io_comm);

      return ierr;
    }
    
  }
}

