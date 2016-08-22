#ifndef POPULATION_READER_HH
#define POPULATION_READER_HH

#include "ngh5types.hh"

#include <map>
#include <set>
#include <map>
#include <vector>

#include "hdf5.h"

extern herr_t read_population_combos
(
 MPI_Comm                             comm,
 const char*                          fname, 
 std::set< std::pair<pop_t,pop_t> >&  pop_pairs
 );

extern herr_t read_population_ranges
(
 MPI_Comm         comm,
 const char*      fname, 
 pop_range_map_t& pop_ranges
 );

/*
extern bool validate_edge_list
(
 const NODE_IDX_T&                         base,
 const std::vector<ROW_PTR_T>&             row_ptr,
 const std::vector<NODE_IDX_T>&            col_idx,
 const pop_range_map_t&                    pop_ranges,
 const std::set< std::pair<pop_t,pop_t> >& pop_pairs
 );
*/

#endif
