// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file validate_edge_list.cc
///
///  Functions for validating edges in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "validate_edge_list.hh"

#include "debug.hh"

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace graph
  {
    bool validate_edge_list
    (
     NODE_IDX_T&         dst_start,
     NODE_IDX_T&         src_start,
     vector<DST_BLK_PTR_T>&  dst_blk_ptr,
     vector<NODE_IDX_T>& dst_idx,
     vector<DST_PTR_T>&  dst_ptr,
     vector<NODE_IDX_T>& src_idx,
     const pop_range_map_t&           pop_ranges,
     const set< pair<pop_t, pop_t> >& pop_pairs
     )
    {
      bool result = true;

      NODE_IDX_T src, dst;

      pop_range_iter_t riter, citer;

      pair<pop_t,pop_t> pp;

      // loop over all edges, look up the node populations, and validate the pairs

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
                      dst = dst_base + ii + dst_start;
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
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
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
                }
            }
        }

      return result;
    }
  }
}
