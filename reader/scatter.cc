
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "dbs_graph_reader.hh"
#include "population_reader.hh"

#include "hdf5.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include <mpi.h>

using namespace std;

/*****************************************************************************
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/


int append_edge_map
(
 const NODE_IDX_T&         base,
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const vector<rank_t>& node_rank_vector,
 rank_edge_map_t & rank_edge_map
 )
{
  int ierr = 0; size_t dst_ptr_size;
  
  if (dst_blk_ptr.size() > 0) 
    {
      dst_ptr_size = dst_ptr.size();
      for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
        {
          size_t low_dst_ptr = dst_blk_ptr[b], high_dst_ptr = dst_blk_ptr[b+1];
          NODE_IDX_T dst_base = base + dst_idx[b];
          for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
            {
              if (i < dst_ptr_size-1) 
                {
                  NODE_IDX_T dst = dst_base + ii + dst_start;
                  // determine the compute rank that the dst node is assigned to
                  rank_t dstrank = node_rank_vector[dst];
                  pair<rank_edge_map_iter_t,bool> r = rank_edge_map.insert(make_pair(dstrank, edge_map_t()));
                  pair<edge_map_iter_t,bool> n = rank_edge_map[dstrank].insert(make_pair(dst, vector<NODE_IDX_T>()));
                  vector<NODE_IDX_T> &v = rank_edge_map[dstrank][dst];
                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  for (size_t j = low; j < high; ++j)
                    {
                      NODE_IDX_T src = src_idx[j] + src_start;
                      v.push_back (src);
                    }
                }
            }
        }
    }

  return ierr;
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  // MPI Communicator for I/O ranks
  MPI_Comm io_comm;
  // MPI group color value used for I/O ranks
  int io_color = 1;
  // A vector that maps nodes to compute ranks
  vector<rank_t> node_rank_vector;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size, io_size; size_t n_nodes;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments

  if (argc < 4) 
    {
      std::cout << "Usage: scatter <FILE> <N> <IOSIZE> [<RANKFILE>]" << std::endl;
      exit(1);
    }

  n_nodes = (size_t)std::stoi(string(argv[2]));
  io_size = std::stoi(string(argv[3]));


  // Am I an I/O rank?
  if (rank < io_size)
    {
      MPI_Comm_split(MPI_COMM_WORLD,io_color,rank,&io_comm);

        // Determine which nodes are assigned to which compute ranks
      node_rank_vector.resize(n_nodes);
      if (argc < 5)
        {
          // round-robin node to rank assignment from file
          for (size_t i = 0; i < n_nodes; i++)
            {
              node_rank_vector[i] = i%size;
            }
        }
      else
        {
          ifstream infile(argv[4]);
          string line;
          size_t i = 0;
          // reads node to rank assignment from file
          while (getline(infile, line))
            {
              istringstream iss(line);
              rank_t n;
              
              assert (iss >> n);
              node_rank_vector[i] = n;
              i++;
            }
          
          infile.close();
        }

      // The set of compute ranks for which the current I/O rank is responsible
      rank_edge_map_t rank_edge_map;
  
      // read the population info
      set< pair<pop_t, pop_t> > pop_pairs;
      assert(read_population_combos(io_comm, argv[1], pop_pairs) >= 0);
      
      vector<pop_range_t> pop_vector;
      map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
      assert(read_population_ranges(io_comm, argv[1], pop_ranges, pop_vector) >= 0);
      
      vector<string> prj_names;
      assert(read_projection_names(io_comm, argv[1], prj_names) >= 0);
      
      
      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          NODE_IDX_T base, dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;
          
          assert(read_dbs_projection(io_comm, argv[1], prj_names[i].c_str(), 
                                     pop_vector, base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
          
          // validate the edges
          assert(validate_edge_list(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, pop_ranges, pop_pairs) == true);
          
          // append to the edge map
          assert(append_edge_map(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, node_rank_vector, rank_edge_map) >= 0);

          for (auto it1 = rank_edge_map.cbegin(); it1 != rank_edge_map.cend(); ++it1)
            {
              printf ("edge_map: it1->first = %u it1->second.size = %lu\n",
                      it1->first, it1->second.size());
              if (it1->second.size() > 0)
                {
                  printf ("edge_map: it2->second keys =");
                  for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                    {
                      printf (" %u", it2->first);
                    }
                  printf("\n");
                }
            }
        }
      
    } else
    {
      MPI_Comm_split(MPI_COMM_WORLD,MPI_UNDEFINED,rank,&io_comm);
    }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_free(&io_comm);
  MPI_Finalize();
  return 0;
}
