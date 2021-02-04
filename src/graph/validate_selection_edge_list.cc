// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file validate_selection_edge_list.cc
///
///  Functions for validating edges in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include "validate_selection_edge_list.hh"

#include "debug.hh"

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace graph
  {
    bool validate_selection_edge_list
    (
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         dst_start,
     const vector<NODE_IDX_T>& selection_dst_idx,
     const vector<DST_PTR_T>&  selection_dst_ptr,
     const vector<NODE_IDX_T>& src_idx,
     const pop_search_range_map_t&    pop_ranges,
     const set< pair<pop_t, pop_t> >& pop_pairs
     )
    {
      bool result = true;

      NODE_IDX_T src, dst;
      pop_search_range_iter_t riter, citer;

      pair<pop_t,pop_t> pp;

      // loop over all edges, look up the node populations, and validate the pairs

      if (selection_dst_idx.size() == 0)
        return result;
      
      size_t dst_ptr_size = selection_dst_ptr.size();
      
      for (size_t i = 0; i < dst_ptr_size-1; ++i)
        {
          dst = selection_dst_idx[i];
          riter = pop_ranges.upper_bound(dst);
          if (riter == pop_ranges.end())
            {
              if (dst <= pop_ranges.rbegin()->first +
                  pop_ranges.rbegin()->second.first)
                {
                  pp.second = pop_ranges.rbegin()->second.second;
                }
              else
                {
                  DEBUG("unable to find population for dst = ",
                        dst,"\n");
                  return false;
                }
            }
          else
            {
              pp.second = riter->second.second-1;
            }
          size_t low = selection_dst_ptr[i], high = selection_dst_ptr[i+1];
          assert((low <= src_idx.size()) && (high <= src_idx.size()));
          if ((high-low) == 0)
            {
              result = true;
            }
          else
            {
              for (size_t j = low; j < high; ++j)
                {
                  src = src_idx[j] + src_start;
                  citer = pop_ranges.upper_bound(src);
                  if (citer == pop_ranges.end())
                    {
                      if (src <= pop_ranges.rbegin()->first +
                          pop_ranges.rbegin()->second.first)
                        {
                          pp.first = pop_ranges.rbegin()->second.second;
                        }
                      else
                        {
                          DEBUG("unable to find population for src = ",src,"\n");
                          return false;
                        }
                    }
                  else
                    {
                      pp.first = citer->second.second-1;
                    }
                  // check if the population combo is valid
                  result = (pop_pairs.find(pp) != pop_pairs.end());
                  if (!result)
                    {
                      DEBUG("invalid edge: src = ",src," dst = ",dst," pp = ",pp.first,", ",pp.second,"\n");
                      return false;
                    }
                }
            }
        }

      return result;
    }
  }
}
