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
    herr_t create_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     const size_t         chunk_size = 1000
     );

    herr_t create_cell_index
    (
     MPI_Comm             comm,
     hid_t                loc,
     const string&        pop_name,
     const string&        attr_name_space,
     const size_t         chunk_size = 1000
     );
    
    herr_t read_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     vector<CELL_IDX_T>&  cell_index
     );

    herr_t append_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const CELL_IDX_T&    pop_start,
     const string&        attr_name_space,
     const vector<CELL_IDX_T>&  cell_index
     );
    
    herr_t append_cell_index
    (
     MPI_Comm             comm,
     hid_t                loc,
     const string&        pop_name,
     const CELL_IDX_T&    pop_start,
     const string&        attr_name_space,
     const vector<CELL_IDX_T>&  cell_index
     );

    herr_t link_cell_index
    (
     MPI_Comm             comm,
     const string&        file_name,
     const string&        pop_name,
     const string&        attr_name_space,
     const string&        attr_name
     );
    
    
  }
}

#endif
