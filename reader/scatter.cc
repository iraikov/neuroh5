
#include "debug.hh"
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
          NODE_IDX_T dst_base = dst_idx[b];
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
  char *input_file_name;
  // MPI Communicator for I/O ranks
  MPI_Comm io_comm, all_comm;
  // MPI group color value used for I/O ranks
  int io_color = 1;
  // A vector that maps nodes to compute ranks
  vector<rank_t> node_rank_vector;
  // The set of compute ranks for which the current I/O rank is responsible
  rank_edge_map_t rank_edge_map;
  set< pair<pop_t, pop_t> > pop_pairs;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  vector<string> prj_names;
  size_t prj_size = 0;
  
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

  input_file_name = argv[1];
  n_nodes = (size_t)std::stoi(string(argv[2]));
  io_size = std::stoi(string(argv[3]));

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);

  // Am I an I/O rank?
  if (rank < io_size)
    {
      MPI_Comm_split(all_comm,io_color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
      
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

  
      // read the population info
      assert(read_population_combos(io_comm, input_file_name, pop_pairs) >= 0);
      assert(read_population_ranges(io_comm, input_file_name, pop_ranges, pop_vector) >= 0);
      assert(read_projection_names(io_comm, input_file_name, prj_names) >= 0);
      prj_size = prj_names.size();
    }
  else
    {
      MPI_Comm_split(all_comm,0,rank,&io_comm);
    }
  MPI_Barrier(all_comm);
  
  assert(MPI_Bcast(&prj_size, 1, MPI_UINT64_T, 0, all_comm) >= 0);
      
  // For each projection, I/O ranks read the edges and scatter
  for (size_t i = 0; i < prj_size; i++)
    {
      size_t num_edges=0;
      vector<int> sendcounts, sdispls, recvcounts, rdispls;
      vector<NODE_IDX_T> edges, recv_edges, total_recv_edges;
      rank_edge_map_t prj_rank_edge_map;

      sendcounts.resize(size,0);
      sdispls.resize(size,0);
      recvcounts.resize(size,0);
      rdispls.resize(size,0);

      if (rank < io_size)
        {
          NODE_IDX_T base, dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;

          assert(read_dbs_projection(io_comm, input_file_name, prj_names[i].c_str(), 
                                     pop_vector, base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
          // validate the edges
          assert(validate_edge_list(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                    pop_ranges, pop_pairs) == true);
          
          // append to the edge map
          assert(append_edge_map(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                 node_rank_vector, prj_rank_edge_map) >= 0);
      
          num_edges = 0;
          for (auto it1 = prj_rank_edge_map.cbegin(); it1 != prj_rank_edge_map.cend(); ++it1)
            {
              uint32_t dst_rank;
              dst_rank = it1->first;
              sdispls[dst_rank] = edges.size();
              if (it1->second.size() > 0)
                {
                  for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                    {
                      NODE_IDX_T dst = it2->first;
                      vector<NODE_IDX_T> vect = it2->second;
                      edges.push_back(dst);
                      sendcounts[dst_rank]++;
                      edges.push_back(vect.size());
                      sendcounts[dst_rank]++;
                      for (size_t j=0; j<vect.size(); j++)
                        {
                          edges.push_back(vect[j]);
                          sendcounts[dst_rank]++;
                          num_edges++;
                        }
                    }
                }
            }
        }

      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
      //    and creates sendcounts and sdispls arrays

      MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT,
                   all_comm);

      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls

      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < size; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      vector<NODE_IDX_T> recvbuf(recvbuf_size);

      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv

      MPI_Alltoallv(&edges[0], &sendcounts[0], &sdispls[0], NODE_IDX_MPI_T,
                    &recvbuf[0], &recvcounts[0], &rdispls[0], NODE_IDX_MPI_T,
                    all_comm);
      edges.clear();

      if (recvbuf.size() > 0) 
        {
          size_t offset = 0;
          ofstream outfile;
          stringstream outfilename;
          outfilename << string(input_file_name) << "." << i << "." << rank << ".edges";
          outfile.open(outfilename.str());
          while (offset < recvbuf.size()-1)
            {
              NODE_IDX_T dst; size_t dst_len;
              dst = recvbuf[offset++];
              dst_len = recvbuf[offset++];
              for (size_t k = 0; k < dst_len; k++)
                {
                  NODE_IDX_T src = recvbuf[offset++];
                  outfile << "    " << src << " " << dst << std::endl;
                }
            }
          outfile.close();
        }

      recvbuf.clear();
    }
  
  MPI_Barrier(all_comm);
  if (rank < io_size)
    {
      MPI_Comm_free(&io_comm);
    }
  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  return 0;
}
