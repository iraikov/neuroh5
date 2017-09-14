// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "debug.hh"
#include "read_projection.hh"
#include "read_projection_datasets.hh"
#include "validate_edge_list.hh"
#include "append_prj_list.hh"

#include <iostream>
#include <sstream>
#include <string>

#undef NDEBUG
#include <cassert>

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
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     vector<prj_tuple_t>&       prj_list,
     size_t&                    local_num_edges,
     size_t&                    total_num_edges,
     size_t                     offset = 0,
     size_t                     numitems = 0,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base, edge_count;
      NODE_IDX_T dst_start, src_start;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;
      map<string, data::NamedAttrVal> edge_attr_map;

      DEBUG("reading projection ", src_pop_name, " -> ", dst_pop_name);
      assert(hdf5::read_projection_datasets(comm, file_name, src_pop_name, dst_pop_name,
                                            dst_start, src_start, block_base, edge_base,
                                            dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                            total_num_edges, offset, numitems) >= 0);
      
      DEBUG("read: validating projection ", i, "(", src_pop_name, " -> ", dst_pop_name, ")");
      
      // validate the edges
      assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                dst_ptr, src_idx, pop_ranges, pop_pairs) ==
             true);
      DEBUG("read: validation of ", i, "(", src_pop_name, " -> ", dst_pop_name, ") finished");
      
      edge_count = src_idx.size();
      local_num_edges = edge_count;
      _
      for (string& attr_namespace : attr_namespaces) 
        {
          vector< pair<string,hid_t> > edge_attr_info;
          
          assert(graph::get_edge_attributes(file_name, src_pop_name, dst_pop_name,
                                            attr_namespace, edge_attr_info) >= 0);
          
          assert(graph::read_all_edge_attributes
                 (comm, file_name, src_pop_name, dst_pop_name, attr_namespace,
                  edge_base, edge_count, edge_attr_info,
                  edge_attr_map[attr_namespace]) >= 0);
        }
      
      size_t local_prj_num_edges=0;
      
      // append to the vectors representing a projection (sources,
      // destinations, edge attributes)
      assert(data::append_prj_list(src_start, dst_start, dst_blk_ptr, dst_idx,
                                   dst_ptr, src_idx, edge_attr_map,
                                   local_prj_num_edges, prj_list) >= 0);
      
      // ensure that all edges in the projection have been read and
      // appended to edge_list
      assert(local_prj_num_edges == edge_count);

      return ierr;
    }

    
    herr_t read_projection_serial
    (
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     vector<prj_tuple_t>&       prj_list,
     size_t&                    total_num_edges,
     size_t                     offset = 0,
     size_t                     numitems = 0
     )
    {
        herr_t ierr = 0;
        DST_PTR_T edge_base, edge_count;
        NODE_IDX_T dst_start, src_start;
        vector<DST_BLK_PTR_T> dst_blk_ptr;
        vector<NODE_IDX_T> dst_idx;
        vector<DST_PTR_T> dst_ptr;
        vector<NODE_IDX_T> src_idx;
        NamedAttrVal edge_attr_values;

        
        assert(hdf5::read_projection_datasets_serial(file_name, src_pop_name, dst_pop_name,
                                                     dst_start, src_start, edge_base,
                                                     dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                                     total_num_edges, offset, numitems) >= 0);

        DEBUG("validating projection ", i, "(", src_pop_name, " -> ", dst_pop_name, ")");
        
        // validate the edges
        assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                  dst_ptr, src_idx, pop_ranges, pop_pairs) ==
               true);
        DEBUG("reader: validation of ", i, "(", src_pop_name, " -> ", dst_pop_name, ") finished");

        edge_count = src_idx.size();
        
        return ierr;
    }
    
  }
}

