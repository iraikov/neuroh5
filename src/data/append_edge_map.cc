
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_edge_map.cc
///
///  Populates a mapping between node indices and edge values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <cassert>
#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "rank_range.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {
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
                                edge_attr_vec.resize<int8_t>
                                  (edge_attr_values.size_attr_vec<int8_t>());
                                edge_attr_vec.resize<int16_t>
                                  (edge_attr_values.size_attr_vec<int16_t>());
                                edge_attr_vec.resize<int32_t>
                                  (edge_attr_values.size_attr_vec<int32_t>());
                                
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
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int8_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int8_t>
                                          (k, edge_attr_values.at<int8_t>(k,j));
                                      }
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int16_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int16_t>
                                          (k, edge_attr_values.at<int16_t>(k,j));
                                      }
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int32_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int32_t>
                                          (k, edge_attr_values.at<int32_t>(k,j));
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
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int8_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int8_t>
                                          (k, edge_attr_values.at<int8_t>(k,j));
                                      }
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int16_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int16_t>
                                          (k, edge_attr_values.at<int16_t>(k,j));
                                      }
                                    for (size_t k = 0;
                                         k < edge_attr_vec.size_attr_vec<int32_t>(); k++)
                                      {
                                        edge_attr_vec.push_back<int32_t>
                                          (k, edge_attr_values.at<int32_t>(k,j));
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
