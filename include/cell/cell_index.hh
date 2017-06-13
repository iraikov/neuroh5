// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_index
///
///  Routines for reading and writing the indices of cells for which attributes are defined.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================
#ifndef CELL_INDEX_HH
#define CELL_INDEX_HH

#include <mpi.h>
#include <vector>
#include "neuroh5_types.hh"

namespace neuroh5
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

    herr_t append_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const vector<CELL_IDX_T>&  cell_index,
     const hsize_t start
     );

  }
}

#endif
