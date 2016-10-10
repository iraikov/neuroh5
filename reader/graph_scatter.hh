#ifndef GRAPH_SCATTER_HH
#define GRAPH_SCATTER_HH

#include "ngh5types.hh"

#include <mpi.h>
#include <hdf5.h>

#include <map>
#include <vector>


int graph_scatter
(
 MPI_Comm all_comm,
 const char *input_file_name,
 const int io_size,
 const bool opt_attrs,
  // A vector that maps nodes to compute ranks
 const std::vector<rank_t> node_rank_vector,
 std::vector < edge_map_t > & prj_vector,
 std::vector < std::vector <uint8_t> > & has_edge_attrs_vector
 );

#endif
