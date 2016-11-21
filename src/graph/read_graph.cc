// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file graph_reader.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "hdf5_path_names.hh"
#include "read_singleton_dataset.hh"
#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "read_graph.hh"
#include "attributes.hh"

#undef NDEBUG
#include <cassert>

using namespace ngh5::io::hdf5;
using namespace ngh5::model;
using namespace std;

namespace ngh5
{
  namespace graph
  {
    int read_graph
    (
     MPI_Comm             comm,
     const std::string&   file_name,
     const bool           opt_attrs,
     const vector<string> prj_names,
     vector<prj_tuple_t>& prj_list,
     size_t&              total_num_nodes,
     size_t&              local_num_edges,
     size_t&              total_num_edges
     )
    {
      // read the population info
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      assert(read_population_combos(comm, file_name, pop_pairs) >= 0);
      assert(read_population_ranges(comm, file_name, pop_ranges, pop_vector,
                                    total_num_nodes) >= 0);

      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          DST_BLK_PTR_T block_base;
          DST_PTR_T edge_base, edge_count;
          NODE_IDX_T dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;
          vector< pair<string,hid_t> > edge_attr_info;
          EdgeNamedAttr edge_attr_values;
          size_t local_prj_num_edges;
          size_t total_prj_num_edges;

          //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

          uint32_t dst_pop, src_pop;

          io::hdf5::read_singleton_dataset
            (comm, file_name, io::hdf5::projection_path_join(prj_names[i],
                                                             io::hdf5::DST_POP),
             H5T_NATIVE_UINT, MPI_UINT32_T, dst_pop);

          io::hdf5::read_singleton_dataset
            (comm, file_name, io::hdf5::projection_path_join(prj_names[i],
                                                             io::hdf5::SRC_POP),
             H5T_NATIVE_UINT, MPI_UINT32_T, src_pop);

          dst_start = pop_vector[dst_pop].start;
          src_start = pop_vector[src_pop].start;

          DEBUG(" dst_start = ", dst_start,
                " src_start = ", src_start,
                "\n");

          assert(io::hdf5::read_dbs_projection
                 (comm, file_name, prj_names[i], dst_start, src_start,
                  total_prj_num_edges, block_base, edge_base, dst_blk_ptr,
                  dst_idx, dst_ptr, src_idx) >= 0);

          DEBUG("reader: projection ", i, " has a total of ",
                total_prj_num_edges, " edges");
          DEBUG("reader: validating projection ", i, "(", prj_names[i], ")");

          // validate the edges
          assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                    dst_ptr, src_idx, pop_ranges, pop_pairs) ==
                 true);
          DEBUG("reader: validation of ", i, "(", prj_names[i], ") finished");

          if (opt_attrs)
            {
              edge_count = src_idx.size();
              assert(get_edge_attributes(file_name, prj_names[i],
                                         edge_attr_info) >= 0);

              assert(read_all_edge_attributes(comm, file_name, prj_names[i],
                                              edge_base, edge_count,
                                              edge_attr_info, edge_attr_values)
                     >= 0);
            }

          DEBUG("reader: ", i, "(", prj_names[i], ") attributes read");

          // append to the vectors representing a projection (sources,
          // destinations, edge attributes)
          assert(append_prj_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                 dst_ptr, src_idx, edge_attr_values,
                                 local_prj_num_edges, prj_list) >= 0);

          // ensure that all edges in the projection have been read and
          // appended to edge_list
          assert(local_prj_num_edges == src_idx.size());

          //printf("Task %d has read %lu edges in projection %lu (%s)\n",
          //       rank,  local_prj_num_edges, i, prj_names[i].c_str());

          total_num_edges = total_num_edges + total_prj_num_edges;
          local_num_edges = local_num_edges + local_prj_num_edges;
        }

      return 0;
    }

    /**************************************************************************
     * Append src/dst node indices to a vector of edges
     **************************************************************************/

    int append_prj_list
    (
     const NODE_IDX_T&                   dst_start,
     const NODE_IDX_T&                   src_start,
     const vector<DST_BLK_PTR_T>&        dst_blk_ptr,
     const vector<NODE_IDX_T>&           dst_idx,
     const vector<DST_PTR_T>&            dst_ptr,
     const vector<NODE_IDX_T>&           src_idx,
     const EdgeNamedAttr&                edge_attr_values,
     size_t&                             num_edges,
     vector<prj_tuple_t>&                prj_list
     )
    {
      int ierr = 0; size_t dst_ptr_size;
      num_edges = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;
      EdgeAttr edge_attr_vec;

      edge_attr_vec.resize<float>
        (edge_attr_values.size_attr_vec<float>());
      edge_attr_vec.resize<uint8_t>
        (edge_attr_values.size_attr_vec<uint8_t>());
      edge_attr_vec.resize<uint16_t>
        (edge_attr_values.size_attr_vec<uint16_t>());
      edge_attr_vec.resize<uint32_t>
        (edge_attr_values.size_attr_vec<uint32_t>());

      if (dst_blk_ptr.size() > 0)
        {
          dst_ptr_size = dst_ptr.size();
          for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
            {
              size_t low_dst_ptr = dst_blk_ptr[b],
                high_dst_ptr = dst_blk_ptr[b+1];

              NODE_IDX_T dst_base = dst_idx[b];
              for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
                {
                  if (i < dst_ptr_size-1)
                    {
                      NODE_IDX_T dst = dst_base + ii + dst_start;
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      for (size_t j = low; j < high; ++j)
                        {
                          NODE_IDX_T src = src_idx[j] + src_start;
                          src_vec.push_back(src);
                          dst_vec.push_back(dst);
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<float>(); k++)
                            {
                              edge_attr_vec.push_back<float>
                                (k, edge_attr_values.at<float>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint8_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint8_t>
                                (k, edge_attr_values.at<uint8_t>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint16_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint16_t>
                                (k, edge_attr_values.at<uint16_t>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint32_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint32_t>
                                (k, edge_attr_values.at<uint32_t>(k,j));
                            }
                          num_edges++;
                        }
                    }
                }
            }
        }

      prj_list.push_back(make_tuple(src_vec, dst_vec, edge_attr_vec));

      return ierr;
    }

    /**************************************************************************
     * Append src/dst node pairs to a list of edges
     **************************************************************************/
    int append_edge_map
    (
     const NODE_IDX_T&             dst_start,
     const NODE_IDX_T&             src_start,
     const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
     const vector<NODE_IDX_T>&     dst_idx,
     const vector<DST_PTR_T>&      dst_ptr,
     const vector<NODE_IDX_T>&     src_idx,
     const EdgeNamedAttr&          edge_attr_values,
     const vector<rank_t>&         node_rank_vector,
     size_t&                       num_edges,
     rank_edge_map_t &             rank_edge_map
     )
    {
      int ierr = 0; size_t dst_ptr_size;

      if (dst_blk_ptr.size() > 0)
        {
          dst_ptr_size = dst_ptr.size();
          for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
            {
              size_t low_dst_ptr = dst_blk_ptr[b],
                high_dst_ptr = dst_blk_ptr[b+1];

              NODE_IDX_T dst_base = dst_idx[b];
              for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
                {
                  if (i < dst_ptr_size-1)
                    {
                      NODE_IDX_T dst = dst_base + ii + dst_start;
                      rank_t dstrank = node_rank_vector[dst];

                      edge_tuple_t& et = rank_edge_map[dstrank][dst];

                      vector<NODE_IDX_T> &my_srcs = get<0>(et);
                      EdgeAttr &edge_attr_vec = get<1>(et);

                      edge_attr_vec.resize<float>
                        (edge_attr_values.size_attr_vec<float>());
                      edge_attr_vec.resize<uint8_t>
                        (edge_attr_values.size_attr_vec<uint8_t>());
                      edge_attr_vec.resize<uint16_t>
                        (edge_attr_values.size_attr_vec<uint16_t>());
                      edge_attr_vec.resize<uint32_t>
                        (edge_attr_values.size_attr_vec<uint32_t>());

                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      for (size_t j = low; j < high; ++j)
                        {
                          NODE_IDX_T src = src_idx[j] + src_start;
                          my_srcs.push_back (src);
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<float>(); k++)
                            {
                              edge_attr_vec.push_back<float>
                                (k, edge_attr_values.at<float>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint8_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint8_t>
                                (k, edge_attr_values.at<uint8_t>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint16_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint16_t>
                                (k, edge_attr_values.at<uint16_t>(k,j));
                            }
                          for (size_t k = 0;
                               k < edge_attr_vec.size_attr_vec<uint32_t>(); k++)
                            {
                              edge_attr_vec.push_back<uint32_t>
                                (k, edge_attr_values.at<uint32_t>(k,j));
                            }

                          num_edges++;
                        }
                    }
                }
            }
        }

      return ierr;
    }

    /**************************************************************************
     * Read edge attributes
     **************************************************************************/

    int read_all_edge_attributes
    (
     MPI_Comm                            comm,
     const string&                       file_name,
     const string&                       prj_name,
     const DST_PTR_T                     edge_base,
     const DST_PTR_T                     edge_count,
     const vector< pair<string,hid_t> >& edge_attr_info,
     EdgeNamedAttr&                      edge_attr_values
     )
    {
      int ierr = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;

      for (size_t j = 0; j < edge_attr_info.size(); j++)
        {
          string attr_name   = edge_attr_info[j].first;
          hid_t  attr_h5type = edge_attr_info[j].second;
          assert ((ierr = read_edge_attributes(comm, file_name, prj_name,
                                               attr_name, edge_base, edge_count,
                                               attr_h5type, edge_attr_values))
                  >= 0);
        }

      return ierr;
    }
  }
}
