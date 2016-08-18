#ifndef CSC_GRAPH_READER_HH
#define CSC_GRAPH_READER_HH

#include "ngh5types.hh"

#include "mpi.h"

#include <vector>

extern herr_t read_csc_graph
(
 MPI_Comm                 comm,
 const char*              fname, 
 NODE_IDX_T&              base,     /* global index of the first node */
 std::vector<BLOCK_PTR_T>&  block_ptr,  
 std::vector<COL_PTR_T>&  col_ptr,  /* one longer than owned nodes count */
 std::vector<NODE_IDX_T>& row_idx
 );

#endif
