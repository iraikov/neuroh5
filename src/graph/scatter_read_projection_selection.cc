// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_projection_selection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "scatter_read_projection_selection.hh"
#include "edge_attributes.hh"
#include "read_projection_dataset_selection.hh"
#include "validate_selection_edge_list.hh"
#include "append_rank_edge_map_selection.hh"
#include "serialize_edge.hh"
#include "serialize_data.hh"
#include "alltoallv_template.hh"
#include "range_sample.hh"
#include "throw_assert.hh"
#include "mpi_debug.hh"

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    /**************************************************************************
     * Reads a subset of the basic DBS graph structure
     *************************************************************************/

    herr_t scatter_read_projection_selection
    (
     MPI_Comm                   comm,
     const int                  io_size,
     const std::string&         file_name,
     const pop_range_map_t&     pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     const vector<string>&      attr_namespaces,
     const std::vector<NODE_IDX_T>&  selection,
     const map<NODE_IDX_T, rank_t>&  node_rank_map,
     vector<edge_map_t>&       prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                    local_num_nodes,
     size_t&                    local_num_edges,
     size_t&                    total_num_edges,
     bool collective
     )
    {
      herr_t ierr = 0;

      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;
      
      int rank, size;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "unable to obtain MPI communicator rank");

      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;

      // Am I an I/O rank?
      if (is_io_rank)
        {
          MPI_Comm_split(comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(comm,0,rank,&io_comm);
        }
      
      local_num_nodes=0; local_num_edges=0;
      rank_edge_map_t prj_rank_edge_map;
      edge_map_t prj_edge_map;
      map<string, vector< vector<string> > > edge_attr_names;

      {
        vector<char> recvbuf;
        vector<int> recvcounts, rdispls;

        {
          vector<char> sendbuf; 
          vector<int> sendcounts(size,0), sdispls(size,0);
          
          if (is_io_rank)
            {
              
              DST_PTR_T edge_base, edge_count;
              vector<NODE_IDX_T> selection_dst_idx;
              vector<DST_PTR_T> selection_dst_ptr;
              vector<NODE_IDX_T> src_idx;
              vector< pair<hsize_t,hsize_t> > src_idx_ranges;
              map<string, data::NamedAttrVal> edge_attr_map;
              hsize_t local_read_blocks;
              
              mpi::MPI_DEBUG(io_comm, "read_projection_selection: ", src_pop_name, " -> ", dst_pop_name, " : "
                             "selection of size ", selection.size());
              throw_assert(hdf5::read_projection_dataset_selection(io_comm, file_name, src_pop_name, dst_pop_name,
                                                                   src_start, dst_start, selection, edge_base,
                                                                   selection_dst_idx, selection_dst_ptr, src_idx_ranges,
                                                                   src_idx, total_num_edges) >= 0,
                           "error in read_projection_dataset_selection");
              
              mpi::MPI_DEBUG(comm, "read_projection_selection: ", src_pop_name, " -> ", dst_pop_name, " :", 
                             " size of destination index is ", selection_dst_idx.size(),
                             " size of destination pointer is ", selection_dst_ptr.size(),
                             "; total_num_edges is ", total_num_edges);
              mpi::MPI_DEBUG(comm, "read_projection_selection: validating projection ", src_pop_name, " -> ", dst_pop_name);
              
              // validate the edges
              throw_assert(validate_selection_edge_list(src_start, dst_start, selection_dst_idx,
                                                        selection_dst_ptr, src_idx, pop_ranges, pop_pairs) ==
                           true,
                           "error in validating edge list"); 
              
              edge_count = src_idx.size();
              local_num_edges = edge_count;

              for (string attr_namespace : attr_namespaces) 
                {
                  vector< pair<string,AttrKind> > edge_attr_info;
                  
                  throw_assert(graph::get_edge_attributes(io_comm, file_name, src_pop_name, dst_pop_name,
                                                          attr_namespace, edge_attr_info) >= 0,
                               "unable to obtain edge attributes");
                  
                  throw_assert(graph::read_all_edge_attribute_selection
                               (io_comm, file_name, src_pop_name, dst_pop_name, attr_namespace,
                                edge_base, edge_count, selection_dst_idx, selection_dst_ptr, src_idx_ranges,
                                edge_attr_info, edge_attr_map[attr_namespace]) >= 0,
                               "error in read_all_edge_attribute_selection");
                  
                  edge_attr_map[attr_namespace].attr_names(edge_attr_names[attr_namespace]);
                }
              
              size_t local_num_prj_edges=0;

              // append to the vectors representing a projection (sources,
              // destinations, edge attributes)
              throw_assert(data::append_rank_edge_map_selection(size, dst_start, src_start, selection_dst_idx, selection_dst_ptr,
                                                                src_idx, attr_namespaces, edge_attr_map, node_rank_map,
                                                                local_num_prj_edges, prj_rank_edge_map,
                                                                EdgeMapDst) >= 0,
                           "error in append_edge_map_selection");

              // ensure that all edges in the projection have been read and
              // appended to edge_list
              throw_assert(local_num_prj_edges == edge_count,
                           "mismatch in projection edge count");
              
              size_t num_packed_edges = 0;
          
              data::serialize_rank_edge_map (size, rank, prj_rank_edge_map, 
                                             num_packed_edges, sendcounts, sendbuf, sdispls);

              // ensure the correct number of edges is being packed
              throw_assert(num_packed_edges == edge_count, "mismatch in packed projection edge count");
              
              mpi::MPI_DEBUG(io_comm, "scatter_read_projection_selection: packed ", num_packed_edges,
                        " edges from projection ", src_pop_name, " -> ", dst_pop_name);

              
            } // is_io_rank

          MPI_Barrier(comm);
          MPI_Barrier(io_comm);
          MPI_Comm_free(&io_comm);
          throw_assert_nomsg(mpi::alltoallv_vector<char>(comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                         recvcounts, rdispls, recvbuf) >= 0);

        }

        if (recvbuf.size() > 0)
          {
            data::deserialize_rank_edge_map (size, recvbuf, recvcounts, rdispls, 
                                             prj_edge_map, local_num_nodes, local_num_edges);
          }

        if (!attr_namespaces.empty())
          {
            vector<char> sendbuf; size_t sendbuf_size=0;
            if (rank == 0)
              {
                data::serialize_data(edge_attr_names, sendbuf);
                sendbuf_size = sendbuf.size();
              }
            
            throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);
            sendbuf.resize(sendbuf_size);
            throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS);
            MPI_Barrier(comm);
            
            if (rank != 0)
              {
                data::deserialize_data(sendbuf, edge_attr_names);
              }
            edge_attr_names_vector.push_back(edge_attr_names);
            
          }
      }
      
      prj_vector.push_back(prj_edge_map);
      
      return ierr;
    }

    
  }
}

