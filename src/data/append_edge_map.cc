
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_edge_map.cc
///
///  Populates a mapping between node indices and edge values.
///
///  Copyright (C) 2016-2025 Project NeuroH5.
//==============================================================================

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
     const NODE_IDX_T&                 dst_start,
     const NODE_IDX_T&                 src_start,
     const vector<DST_BLK_PTR_T>&      dst_blk_ptr,
     const vector<NODE_IDX_T>&         dst_idx,
     const vector<DST_PTR_T>&          dst_ptr,
     const vector<NODE_IDX_T>&         src_idx,
     const vector<string>&             attr_namespaces,
     const map<string, NamedAttrVal>&  edge_attr_map,
     size_t&                           num_edges,
     edge_map_t &                      edge_map,
     EdgeMapType                       edge_map_type
     )
    {
      int ierr = 0; size_t dst_ptr_size;
      
      // Ensure we have data to process
      if (dst_blk_ptr.empty() || dst_idx.empty() || dst_ptr.empty() || src_idx.empty())
        {
          return ierr;
        };

      if (dst_blk_ptr.size() > 0)
        {
          dst_ptr_size = dst_ptr.size();
          for (size_t b = 0; b < dst_idx.size(); ++b)
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

                                vector <AttrVal> &edge_attr_vec = get<1>(et);
                                edge_attr_vec.resize(edge_attr_map.size());
                                
                                for (size_t j = low; j < high; ++j)
                                  {
                                    NODE_IDX_T src = src_idx[j] + src_start;
                                    my_srcs.push_back (src);
                                    num_edges++;
                                  }
                                
                                fill_attr_vec<float>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<uint8_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<uint16_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<uint32_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<int8_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<int16_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                                fill_attr_vec<int32_t>(edge_attr_map, attr_namespaces, edge_attr_vec, low, high);
                              }
                              break;
                            case EdgeMapSrc:
                              {
                                for (size_t j = low; j < high; ++j)
                                  {
                                    NODE_IDX_T src = src_idx[j] + src_start;

                                    edge_tuple_t& et = edge_map[src];

                                    vector<NODE_IDX_T> &my_dsts = get<0>(et);

                                    vector <AttrVal> &edge_attr_vec = get<1>(et);
                                    edge_attr_vec.resize(edge_attr_map.size());

                                    my_dsts.push_back(dst);
                                    
                                    set_attr_vec<float>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<uint8_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<uint16_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<uint32_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<int8_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<int16_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);
                                    set_attr_vec<int32_t>(edge_attr_map, attr_namespaces, edge_attr_vec, j);

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
