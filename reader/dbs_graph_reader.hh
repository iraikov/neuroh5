#ifndef DBS_GRAPH_READER_HH
#define DBS_GRAPH_READER_HH

#include "ngh5types.hh"

#include "mpi.h"

#include <vector>

extern herr_t read_dbs_graph
(
 MPI_Comm                 comm,
 const char*              fname, 
 const char*              dsetname, 
 NODE_IDX_T&              base,     /* global index of the first node */
 std::vector<DST_BLK_PTR_T>&  dst_blk_ptr,  
 std::vector<NODE_IDX_T>& dst_idx,
 std::vector<DST_PTR_T>&  dst_ptr,  /* one longer than owned nodes count */
 std::vector<NODE_IDX_T>& src_idx
 );

#endif
