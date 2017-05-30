// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_tree_index
///
///  
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================
#ifndef READ_TREE_INDEX_HH
#define READ_TREE_INDEX_HH

#include <mpi.h>
#include <vector>
#include "neurotrees_types.hh"

namespace neurotrees
{
  herr_t read_tree_index
  (
   MPI_Comm             comm,
   const string&        file_name,
   const string&        pop_name,
   vector<CELL_IDX_T>&  tree_index
   );

}

#endif
