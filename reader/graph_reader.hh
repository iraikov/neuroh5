#ifndef GRAPH_SCATTER_HH
#define GRAPH_SCATTER_HH

#include "ngh5types.hh"

#include <mpi.h>
#include <hdf5.h>

#include <map>
#include <vector>

namespace ngh5
{

int read_graph
(
 MPI_Comm comm,
 const char *input_file_name,
 const bool opt_attrs,
 const std::vector<std::string> prj_names,
 std::vector<prj_tuple_t> &prj_list,
 size_t &local_prj_num_edges,
 size_t &total_prj_num_edges
 );

int scatter_graph
(
 MPI_Comm all_comm,
 const char *input_file_name,
 const int io_size,
 const bool opt_attrs,
 const std::vector<std::string> prj_names,
  // A vector that maps nodes to compute ranks
 const std::vector<rank_t> node_rank_vector,
 std::vector < edge_map_t > & prj_vector
 );

  
}

#endif
