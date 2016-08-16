#ifndef CRS_GRAPH_READER_HH
#define CRS_GRAPH_READER_HH

#include "ngh5types.h"

#include "mpi.h"

#include <vector>

extern herr_t read_crs_graph
(
MPI_Comm                 comm,
const char*              fname, 
NODE_IDX_T&              base,     /* global index of the first node */
std::vector<ROW_PTR_T>&  row_ptr,  /* one longer than owned nodes count */
std::vector<NODE_IDX_T>& col_idx
);

#endif
