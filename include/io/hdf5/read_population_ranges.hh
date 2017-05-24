// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_population_ranges.hh
///
///  
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================
#ifndef READ_POPULATION_RANGES_HH
#define READ_POPULATION_RANGES_HH

#include <mpi.h>
#include <vector>
#include <map>
#include "neurotrees_types.hh"

namespace neurotrees
{

 herr_t read_population_ranges
 (
  MPI_Comm                  comm,
  const std::string&        file_name,
  std::map<CELL_IDX_T, std::pair<uint32_t,pop_t> >& pop_ranges,
  std::vector<pop_range_t>& pop_vector,
  size_t& n_nodes
  );
}

#endif
