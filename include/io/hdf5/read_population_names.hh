// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_population_names
///
///  
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================
#ifndef READ_POPULATION_NAMES_HH
#define READ_POPULATION_NAMES_HH

#include <mpi.h>
#include <vector>
#include "neurotrees_types.hh"

namespace neurotrees
{
  herr_t read_population_names
  (
   MPI_Comm             comm,
   hid_t                file,
   vector<string>&      pop_names
   );

}

#endif
