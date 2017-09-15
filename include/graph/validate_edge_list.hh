#ifndef VALIDATE_EDGE_LIST_HH
#define VALIDATE_EDGE_LIST_HH

#include "neuroh5_types.hh"

#include <set>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    /// @brief Validates that each edge in a projection has source and
      ///        destination indices that are within the population ranges
      ///        defined for that projection
      ///
      /// @param dst_start     Updated with global starting index of destination
      ///                      population
      ///
      /// @param src_start     Updated with global starting index of source
      ///                      population
      ///
      /// @param dst_blk_ptr   Destination Block Pointer (pointer to Destination
      ///                      Pointer for blocks of connectivity)
      ///
      /// @param dst_idx       Destination Index (starting destination index for
      ///                      each block)
      ///
      /// @param dst_ptr       Destination Pointer (pointer to Source Index
      ///                      where the source indices for a given destination
      ///                      can be located)
      ///
      /// @param src_idx       Source Index (source indices of edges)
      ///
      /// @param pop_ranges    Map where the key is the starting index of a
      ///                      population, and the value is the number of nodes
      ///                      (vertices) and population index, filled by this
      ///                      procedure
      ///
      /// @param pop_pairs     Set of source/destination pairs, filled by this
      ///                      procedure
      ///
      /// @return              True if the edges are valid, false otherwise
      extern bool validate_edge_list
      (
       const NODE_IDX_T&                                              dst_start,
       const NODE_IDX_T&                                              src_start,
       const std::vector<DST_BLK_PTR_T>&                              dst_blk_ptr,
       const std::vector<NODE_IDX_T>&                                 dst_idx,
       const std::vector<DST_PTR_T>&                                  dst_ptr,
       const std::vector<NODE_IDX_T>&                                 src_idx,
       const pop_range_map_t&                                   pop_ranges,
       const std::set< std::pair<pop_t, pop_t> >&               pop_pairs
       );
  }
}

#endif
