// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_projection.hh
///
///  Top-level functions for reading projections in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================


#ifndef SCATTER_READ_PROJECTION_HH
#define SCATTER_READ_PROJECTION_HH

#include "neuroh5_types.hh"

#include <mpi.h>

#include <map>
#include <vector>

namespace neuroh5
{
  namespace graph
  {

    int scatter_read_projection (MPI_Comm all_comm,
                                 const int io_size,
                                 const EdgeMapType edge_map_type, 
                                 const string& file_name,
                                 const string& src_pop_name,
                                 const string& dst_pop_name, 
                                 const std::vector< std::string >&  attr_namespaces,
                                 const node_rank_map_t&  node_rank_map,
                                 const std::vector<pop_range_t>& pop_vector,
                                 const std::map<NODE_IDX_T, std::pair<uint32_t,pop_t> >& pop_ranges,
                                 const std::vector< std::pair<pop_t, string> >& pop_labels,
                                 const std::set< std::pair<pop_t, pop_t> >& pop_pairs,
                                 std::vector < edge_map_t >& prj_vector,
                                 std::vector < map <string, std::vector < std::vector<string> > > > & edge_attr_names_vector,
                                 size_t &local_num_nodes, size_t &local_num_edges, size_t &total_num_edges,
                                 hsize_t &total_read_blocks,
                                 size_t offset = 0, size_t numitems = 0);
  }
}

#endif
