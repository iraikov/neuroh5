#ifndef POPULATION_READER_HH
#define POPULATION_READER_HH

#include "ngh5types.h"

#include <map>
#include <set>
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
MPI_Comm                   comm,
const char*                fname, 
std::vector<pop_range_t>&  pop_ranges
);

#endif
