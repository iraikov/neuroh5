
// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file append_prj_list.cc
///
///  Populates a projection vector with edge values.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <cassert>
#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_val.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"

using namespace std;

namespace neuroh5
{

  namespace data
  {

    
    /**************************************************************************
     * Append src/dst node indices to a vector of edges
     **************************************************************************/

    int append_prj_list
    (
     const NODE_IDX_T&                   src_start,
     const NODE_IDX_T&                   dst_start,
     const vector<DST_BLK_PTR_T>&        dst_blk_ptr,
     const vector<NODE_IDX_T>&           dst_idx,
     const vector<DST_PTR_T>&            dst_ptr,
     const vector<NODE_IDX_T>&           src_idx,
     const map<string, NamedAttrVal>&    edge_attr_map,
     size_t&                             num_edges,
     vector<prj_tuple_t>&                prj_list
     )
    {
      int ierr = 0; size_t dst_ptr_size;
      num_edges = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;
      vector<AttrVal> edge_attr_vec(edge_attr_map.size());

      size_t i=0;
      for (auto item : edge_attr_map) 
        {
          const string & attr_namespace = item->first;
          const NamedAttrVal& edge_attr_values = item->second;
          
          edge_attr_vec[i].resize<float>
            (edge_attr_values.size_attr_vec<float>());
          edge_attr_vec[i].resize<uint8_t>
            (edge_attr_values.size_attr_vec<uint8_t>());
          edge_attr_vec[i].resize<uint16_t>
            (edge_attr_values.size_attr_vec<uint16_t>());
          edge_attr_vec[i].resize<uint32_t>
            (edge_attr_values.size_attr_vec<uint32_t>());
          edge_attr_vec[i].resize<int8_t>
            (edge_attr_values.size_attr_vec<int8_t>());
          edge_attr_vec[i].resize<int16_t>
            (edge_attr_values.size_attr_vec<int16_t>());
          edge_attr_vec[i].resize<int32_t>
            (edge_attr_values.size_attr_vec<int32_t>());
          
          i++;
        }

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
                          fill_attr_vec<float>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<uint8_t>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<uint16_t>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<uint32_t>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<int8_t>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<int16_t>(edge_attr_map, edge_attr_vec, j);
                          fill_attr_vec<int32_t>(edge_attr_map, edge_attr_vec, j);
                          num_edges++;
                        }
                    }
                }
            }
        }

      prj_list.push_back(make_tuple(src_vec, dst_vec, edge_attr_vec));

      return ierr;
    }
    
  }
}
