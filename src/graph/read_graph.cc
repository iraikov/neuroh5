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

#include "read_dbs_projection.hh"
#include "cell_populations.hh"
#include "read_graph.hh"
#include "read_population.hh"
#include "validate_edge_list.hh"

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
     MPI_Comm             comm,
     const std::string&   file_name,
     const bool           opt_attrs,
     const vector< pair<string, string> > prj_names,
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
      assert(cell::read_population_combos(comm, file_name, pop_pairs) >= 0);
      assert(cell::read_population_ranges
             (comm, file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);
      assert(cell::read_population_labels(comm, file_name, pop_labels) >= 0);

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
          NamedAttrVal edge_attr_values;
          size_t local_prj_num_edges;
          size_t total_prj_num_edges;

          //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

          string src_pop_name = prj_names[i].first, dst_pop_name = prj_names[i].second;
          uint32_t dst_pop_idx, src_pop_idx;
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

          assert(graph::read_projection
                 (comm, file_name, src_pop_name, dst_pop_name, dst_start, src_start,
                  total_prj_num_edges, block_base, edge_base, dst_blk_ptr,
                  dst_idx, dst_ptr, src_idx) >= 0);

          DEBUG("reader: projection ", i, " has a total of ",
                total_prj_num_edges, " edges");
          DEBUG("reader: validating projection ", i, "(", src_pop_name, " -> ", dst_pop_name ")");

          // validate the edges
          assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                    dst_ptr, src_idx, pop_ranges, pop_pairs) ==
                 true);
          DEBUG("reader: validation of ", i, "(", src_pop_name, " -> ", dst_pop_name, ") finished");

          if (opt_attrs)
            {
              edge_count = src_idx.size();
              assert(graph::get_edge_attributes(file_name, src_pop_name, dst_pop_name,
                                                edge_attr_info) >= 0);

              assert(graph::read_all_edge_attributes
                     (comm, file_name, src_pop_name, dst_pop_name, edge_base, edge_count,
                      edge_attr_info, edge_attr_values) >= 0);
            }

          DEBUG("reader: ", i, "(", src_pop_name, " -> ", dst_pop_name, ") attributes read");

          // append to the vectors representing a projection (sources,
          // destinations, edge attributes)
          assert(append_prj_list(dst_start, src_start, dst_blk_ptr, dst_idx,
                                 dst_ptr, src_idx, edge_attr_values,
                                 local_prj_num_edges, prj_list) >= 0);

          // ensure that all edges in the projection have been read and
          // appended to edge_list
          assert(local_prj_num_edges == src_idx.size());

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
     const NamedAttrVal&                 edge_attr_values,
     size_t&                             num_edges,
     vector<prj_tuple_t>&                prj_list
     )
    {
      int ierr = 0; size_t dst_ptr_size;
      num_edges = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;
      AttrVal edge_attr_vec;

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
     * Append src/dst node pairs to a map of edges
     **************************************************************************/
    int append_edge_map
    (
     const NODE_IDX_T&             dst_start,
     const NODE_IDX_T&             src_start,
     const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
     const vector<NODE_IDX_T>&     dst_idx,
     const vector<DST_PTR_T>&      dst_ptr,
     const vector<NODE_IDX_T>&     src_idx,
     const NamedAttrVal&           edge_attr_values,
     size_t&                       num_edges,
     edge_map_t &                  edge_map,
     EdgeMapType                   edge_map_type
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
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      if (high > low)
                        {
                          switch (edge_map_type)
                            {
                            case EdgeMapDst:
                              {
                                edge_tuple_t& et = edge_map[dst];
                                vector<NODE_IDX_T> &my_srcs = get<0>(et);

                                AttrVal &edge_attr_vec = get<1>(et);
                      
                                edge_attr_vec.resize<float>
                                  (edge_attr_values.size_attr_vec<float>());
                                edge_attr_vec.resize<uint8_t>
                                  (edge_attr_values.size_attr_vec<uint8_t>());
                                edge_attr_vec.resize<uint16_t>
                                  (edge_attr_values.size_attr_vec<uint16_t>());
                                edge_attr_vec.resize<uint32_t>
                                  (edge_attr_values.size_attr_vec<uint32_t>());
                                
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
                              break;
                            case EdgeMapSrc:
                              {
                                for (size_t j = low; j < high; ++j)
                                  {
                                    NODE_IDX_T src = src_idx[j] + src_start;

                                    edge_tuple_t& et = edge_map[src];

                                    vector<NODE_IDX_T> &my_dsts = get<0>(et);

                                    AttrVal &edge_attr_vec = get<1>(et);
                                    
                                    edge_attr_vec.resize<float>
                                      (edge_attr_values.size_attr_vec<float>());
                                    edge_attr_vec.resize<uint8_t>
                                      (edge_attr_values.size_attr_vec<uint8_t>());
                                    edge_attr_vec.resize<uint16_t>
                                      (edge_attr_values.size_attr_vec<uint16_t>());
                                    edge_attr_vec.resize<uint32_t>
                                      (edge_attr_values.size_attr_vec<uint32_t>());

                                    my_dsts.push_back(dst);
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
                              break;
                            }
                        }
                    }
                }
            }
        }

      return ierr;
    }

    
    /**************************************************************************
     * Append src/dst node pairs to a map of ranks and edges
     **************************************************************************/
    int append_rank_edge_map
    (
     const NODE_IDX_T&             dst_start,
     const NODE_IDX_T&             src_start,
     const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
     const vector<NODE_IDX_T>&     dst_idx,
     const vector<DST_PTR_T>&      dst_ptr,
     const vector<NODE_IDX_T>&     src_idx,
     const NamedAttrVal&           edge_attr_values,
     const map<NODE_IDX_T, rank_t>&  node_rank_map,
     size_t&                       num_edges,
     rank_edge_map_t &             rank_edge_map,
     EdgeMapType                   edge_map_type
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
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      if (high > low)
                        {
                          switch (edge_map_type)
                            {
                            case EdgeMapDst:
                              {
                                auto it = node_rank_map.find(dst);
                                if (it == node_rank_map.end())
                                  {
                                    printf("gid %d not found in rank map\n", dst);
                                  }
                                //assert(it != node_rank_map.end());
                                rank_t myrank;
                                if (it == node_rank_map.end())
                                { myrank = 0; }
                                else
                                { myrank = it->second; }
                                edge_tuple_t& et = rank_edge_map[myrank][dst];
                                vector<NODE_IDX_T> &my_srcs = get<0>(et);

                                AttrVal &edge_attr_vec = get<1>(et);
                      
                                edge_attr_vec.resize<float>
                                  (edge_attr_values.size_attr_vec<float>());
                                edge_attr_vec.resize<uint8_t>
                                  (edge_attr_values.size_attr_vec<uint8_t>());
                                edge_attr_vec.resize<uint16_t>
                                  (edge_attr_values.size_attr_vec<uint16_t>());
                                edge_attr_vec.resize<uint32_t>
                                  (edge_attr_values.size_attr_vec<uint32_t>());
                                
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
                              break;
                            case EdgeMapSrc:
                              {
                                for (size_t j = low; j < high; ++j)
                                  {
                                    NODE_IDX_T src = src_idx[j] + src_start;

                                    auto it = node_rank_map.find(src);
                                    assert(it != node_rank_map.end());
                                    rank_t myrank = it->second;
                                    edge_tuple_t& et = rank_edge_map[myrank][src];

                                    vector<NODE_IDX_T> &my_dsts = get<0>(et);

                                    AttrVal &edge_attr_vec = get<1>(et);
                                    
                                    edge_attr_vec.resize<float>
                                      (edge_attr_values.size_attr_vec<float>());
                                    edge_attr_vec.resize<uint8_t>
                                      (edge_attr_values.size_attr_vec<uint8_t>());
                                    edge_attr_vec.resize<uint16_t>
                                      (edge_attr_values.size_attr_vec<uint16_t>());
                                    edge_attr_vec.resize<uint32_t>
                                      (edge_attr_values.size_attr_vec<uint32_t>());

                                    my_dsts.push_back(dst);
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
                              break;
                            }
                        }
                    }
                }
            }
        }

      return ierr;
    }
  }
}
