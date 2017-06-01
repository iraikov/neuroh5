// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_cell_index
///
///  
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================
#ifndef READ_CELL_INDEX_HH
#define READ_CELL_INDEX_HH

#include <mpi.h>
#include <vector>
#include "neuroio_types.hh"

namespace neuroio
{
  namespace cell
  {
    
    herr_t read_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     vector<CELL_IDX_T>&  cell_index
     );

  }
}

#endif
