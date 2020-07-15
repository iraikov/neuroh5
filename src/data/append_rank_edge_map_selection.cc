// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_rank_edge_map_selection.cc
///
///  Populates a mapping between node indices and edge values.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
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
    int append_rank_edge_map_selection
    (
     const size_t                      num_ranks,
     const NODE_IDX_T&                 dst_start,
     const NODE_IDX_T&                 src_start,
     const vector<NODE_IDX_T>&         selection_dst_idx,
     const vector<DST_PTR_T>&          selection_dst_ptr,
     const vector<NODE_IDX_T>&         src_idx,
     const vector<string>&             attr_namespaces,
     const map<string, NamedAttrVal>&  edge_attr_map,
     const map<NODE_IDX_T, rank_t>&    node_rank_map,
     size_t&                           num_edges,
     rank_edge_map_t &                 rank_edge_map,
     EdgeMapType                       edge_map_type
     )
    {
      int ierr = 0; size_t dst_ptr_size;

      if (selection_dst_idx.size() == 0)
        return ierr;
      
      dst_ptr_size = selection_dst_ptr.size();

      for (size_t i = 0; i < dst_ptr_size-1; ++i)
        {
          size_t num_dst = i;

          NODE_IDX_T dst = selection_dst_idx[i];
          size_t low = selection_dst_ptr[i], high = selection_dst_ptr[i+1];
          if (high > low)
            {
              switch (edge_map_type)
                {
                case EdgeMapDst:
                  {
                    auto it = node_rank_map.find(dst);
                    rank_t myrank=0;
                    if (it == node_rank_map.end())
                      { myrank = num_dst % num_ranks; }
                    else
                      { myrank = it->second; }

                    edge_tuple_t& et = rank_edge_map[myrank][dst];
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
                        rank_t myrank = 0;
                        auto it = node_rank_map.find(src);
                        if (it == node_rank_map.end())
                          { myrank = src % num_ranks; }
                        else
                          { myrank = it->second; }
                        
                        edge_tuple_t& et = rank_edge_map[myrank][src];
                        
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

      return ierr;
    }
  }
}
