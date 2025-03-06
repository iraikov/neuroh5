// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file validate_edge_list.cc
///
///  Functions for validating edges in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2022 Project NeuroH5.
//==============================================================================

#include "validate_edge_list.hh"
#include "throw_assert.hh"
#include "debug.hh"

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace graph
  {
    bool validate_edge_list
    (
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         src_start,
     const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
     const vector<NODE_IDX_T>& dst_idx,
     const vector<DST_PTR_T>&  dst_ptr,
     const vector<NODE_IDX_T>& src_idx,
     const pop_search_range_map_t&           pop_search_ranges,
     const set< pair<pop_t, pop_t> >& pop_pairs
     )
    {
      bool result = true;

      NODE_IDX_T src, dst;

      pop_search_range_iter_t riter, citer;

      pair<pop_t,pop_t> pp;

      // loop over all edges, look up the node populations, and validate the pairs
      if (dst_idx.size() > 0)
        {
          size_t dst_ptr_size = dst_ptr.size();
          for (size_t b = 0; b < dst_idx.size(); ++b)
            {
              size_t low_dst_ptr = dst_blk_ptr[b],
                high_dst_ptr = dst_blk_ptr[b+1];

              NODE_IDX_T dst_base = dst_idx[b];
              for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
                {
                  if (i < dst_ptr_size-1)
                    {
                      dst = dst_base + ii + dst_start;
                      riter = pop_search_ranges.upper_bound(dst);
                      if (riter == pop_search_ranges.end())
                        {
                          if (dst <= pop_search_ranges.rbegin()->first +
                              pop_search_ranges.rbegin()->second.first)
                            {
                              pp.second = pop_search_ranges.rbegin()->second.second;
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
                          if (riter != pop_search_ranges.begin())
                            {
                              --riter;
                              pp.second = riter->second.second;
                            }
                          else
                            {
                              DEBUG("unable to find population for dst = ",
                                    dst,"\n");
                              return false;
                            }
                        }
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      throw_assert_nomsg((low <= src_idx.size()) && (high <= src_idx.size()));
                      if ((high-low) == 0)
                        {
                          result = true;
                        }
                      else
                        {
                          for (size_t j = low; j < high; ++j)
                            {
                              src = src_idx[j] + src_start;
                              citer = pop_search_ranges.upper_bound(src);
                              if (citer == pop_search_ranges.end())
                                {
                                  if (src <= pop_search_ranges.rbegin()->first +
                                      pop_search_ranges.rbegin()->second.first)
                                    {
                                      pp.first = pop_search_ranges.rbegin()->second.second;
                                    }
                                  else
                                    {
                                      DEBUG("unable to find population for src = ", src,"\n");
                                      return false;
                                    }
                                }
                              else
                                {
                                  if (citer != pop_search_ranges.begin())
                                    {
                                      --citer;
                                      pp.first = citer->second.second;
                                    }
                                  else
                                    {
                                      DEBUG("unable to find population for src = ", src,"\n");
                                      return false;
                                    }
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
