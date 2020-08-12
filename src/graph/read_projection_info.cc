// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_info.cc
///
///  Functions for reading edge information.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection.hh"
#include "edge_attributes.hh"
#include "read_projection_datasets.hh"
#include "validate_edge_list.hh"
#include "append_edge_map.hh"
#include "mpi_debug.hh"

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
     * Reads information about the given projection.
     *************************************************************************/

    herr_t read_projection_info
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const vector<string>& edge_attr_name_spaces,
     const bool                 read_node_index,
     const pop_range_map_t&     pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     vector< pair<string, string> >& prj_names,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     std::vector<std::vector<NODE_IDX_T>>& prj_node_index
     
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");
      throw_assert(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");

      
      
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;

      map<string, data::NamedAttrVal> edge_attr_map;
      std::vector<NODE_IDX_T> node_index;
      if (read_node_index)
        {
          int io_color = 1;
          MPI_Comm io_comm;

          if (rank == 0)
            {
              MPI_Comm_split(comm,io_color,rank,&io_comm);
              MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
            }
          else
            {
              MPI_Comm_split(comm,0,rank,&io_comm);
            }

          if (rank == 0)
            {
              throw_assert(hdf5::read_projection_node_datasets(io_comm, file_name, src_pop_name, dst_pop_name,
                                                               dst_start, src_start, block_base, edge_base,
                                                               dst_blk_ptr, dst_idx, dst_ptr) >= 0,
                           "read_projection_info: read_projection_node_datasets error");
            }
          MPI_Barrier(comm);
          MPI_Comm_free(&io_comm);
          
          if (dst_blk_ptr.size() > 0)
            {
              size_t dst_ptr_size = dst_ptr.size();
              for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
                {
                  size_t low_dst_ptr = dst_blk_ptr[b],
                    high_dst_ptr = dst_blk_ptr[b+1];
                  
                  NODE_IDX_T dst_base = dst_idx[b];
                  for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
                    {
                      if (i < dst_ptr_size-1)
                        {
                          NODE_IDX_T node = dst_base + ii + dst_start;
                          node_index.push_back(node);
                        }
                    }
                }
            }

          prj_node_index.push_back(node_index);
        }
      

      map <string, vector < vector<string> > > edge_attr_names;
      for (string attr_namespace : edge_attr_name_spaces) 
        {
          bool has_namespace = false;
          throw_assert(has_edge_attribute_namespace(comm, file_name, src_pop_name, dst_pop_name, attr_namespace, has_namespace),
                       "read_projection: has_edge_attribute_namespace error");
          if (has_namespace)
            {
              vector< pair<string,AttrKind> > edge_attr_info;
              
              throw_assert(graph::get_edge_attributes(comm, file_name, src_pop_name, dst_pop_name,
                                                      attr_namespace, edge_attr_info) >= 0,
                           "read_projection: get_edge_attributes error");
              
              edge_attr_map[attr_namespace].attr_names(edge_attr_names[attr_namespace]);
            }
        }
      
      size_t local_prj_num_edges=0;

      edge_attr_names_vector.push_back (edge_attr_names);

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "read_projection_info: error in MPI_Barrier");

      
      return ierr;
    }

    
  }
}

