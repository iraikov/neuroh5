// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================


#include "neuroh5_types.hh"
#include "read_projection.hh"
#include "edge_attributes.hh"
#include "read_projection_datasets.hh"
#include "validate_edge_list.hh"
#include "append_edge_map.hh"
#include "mpi_debug.hh"
#include "debug.hh"

#include <iostream>
#include <sstream>
#include <string>

#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    /**************************************************************************
     * Read the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const pop_range_map_t&     pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     const vector<string>&      attr_namespaces,
     vector<edge_map_t>&       prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                    local_num_nodes,
     size_t&                    local_num_edges,
     size_t&                    total_num_edges,
     hsize_t&                   local_read_blocks,
     hsize_t&                   total_read_blocks,
     size_t                     offset,
     size_t                     numitems,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");
      throw_assert(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");

      
      
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base, edge_count;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;
      map<string, data::NamedAttrVal> edge_attr_map;

      mpi::MPI_DEBUG(comm, "read_projection: ", src_pop_name, " -> ", dst_pop_name);
      throw_assert(hdf5::read_projection_datasets(comm, file_name, src_pop_name, dst_pop_name,
                                                  dst_start, src_start, block_base, edge_base,
                                                  dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                  total_num_edges, total_read_blocks, local_read_blocks,
                                                  offset, numitems) >= 0,
                   "read_projection: read_projection_datasets error");
      
      mpi::MPI_DEBUG(comm, "read_projection: validating projection ", src_pop_name, " -> ", dst_pop_name);
      
      // validate the edges
      throw_assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                      dst_ptr, src_idx, pop_ranges, pop_pairs) ==
                   true, "read_projection: invalid edge list");
      
      edge_count = src_idx.size();
      local_num_edges = edge_count;

      map <string, vector < vector<string> > > edge_attr_names;
      for (string attr_namespace : attr_namespaces) 
        {
          vector< pair<string,AttrKind> > edge_attr_info;
          
          throw_assert(graph::get_edge_attributes(comm, file_name, src_pop_name, dst_pop_name,
                                                  attr_namespace, edge_attr_info) >= 0,
                       "read_projection: get_edge_attributes error");
          
          throw_assert(graph::read_all_edge_attributes
                       (comm, file_name, src_pop_name, dst_pop_name, attr_namespace,
                        edge_base, edge_count, edge_attr_info,
                        edge_attr_map[attr_namespace]) >= 0,
                       "read_projection: read_all_edge_attributes error");
          
          edge_attr_map[attr_namespace].attr_names(edge_attr_names[attr_namespace]);
        }
      
      size_t local_prj_num_edges=0;

      edge_map_t prj_edge_map;
      // append to the vectors representing a projection (sources,
      // destinations, edge attributes)
      throw_assert(data::append_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx,
                                         dst_ptr, src_idx, attr_namespaces, edge_attr_map,
                                         local_prj_num_edges, prj_edge_map,
                                         EdgeMapDst) >= 0,
                   "read_projection: error in append_edge_map");
      local_num_nodes = prj_edge_map.size();
      
      // ensure that all edges in the projection have been read and
      // appended to edge_list
      throw_assert(local_prj_num_edges == edge_count,
                   "read_projection: edge count mismatch");

      prj_vector.push_back(prj_edge_map);
      edge_attr_names_vector.push_back (edge_attr_names);
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "read_projection: error in MPI_Barrier");
#endif
      
      return ierr;
    }

    
  }
}

