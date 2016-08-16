#ifndef POPULATION_READER_HH
#define POPULATION_READER_HH

#include "ngh5types.h"

#include <map>
#include <set>

#include "hdf5.h"

extern herr_t read_population_combos
(
MPI_Comm                             comm,
const char*                          fname, 
std::set< std::pair<pop_t,pop_t> >&  pop_pairs
);

#endif
