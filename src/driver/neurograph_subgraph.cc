#include "debug.hh"

#include "neuroh5_types.hh"
#include "cell_populations.hh"

#include "read_projection.hh"
#include "read_graph.hh"
#include "scatter_graph.hh"
#include "cell_populations.hh"
#include "projection_names.hh"
#include "validate_edge_list.hh"

#include "hdf5.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include <mpi.h>

using namespace std;
using namespace neuroh5;

/*****************************************************************************
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/

int filter_edge_list
(
 const NODE_IDX_T&         base,
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const set<NODE_IDX_T>&    src_selection,
 const set<NODE_IDX_T>&    dst_selection,
 vector<NODE_IDX_T>&       edge_list
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
                  if (dst_selection.empty() || (dst_selection.find(dst) != dst_selection.end()))
                    {
                      size_t low = dst_ptr[i], high = dst_ptr[i+1];
                      for (size_t j = low; j < high; ++j)
                        {
                          NODE_IDX_T src = src_idx[j] + src_start;
                          if (src_selection.empty() || (src_selection.find(src) != src_selection.end()))
                            {
                              edge_list.push_back(src);
                              edge_list.push_back(dst);
                            }
                        }
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
  string input_file_name, src_pop_name, dst_pop_name;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments

  if (argc < 3) 
    {
      std::cout << "Usage: reader <FILE> <SRC> <DST> <SELECTION> ..." << std::endl;
      exit(1);
    }

  input_file_name = string(argv[1]);
  src_pop_name = string(argv[2]);
  dst_pop_name = string(argv[3]);

  // determine src and dst node selections
  set <NODE_IDX_T> src_selection;
  set <NODE_IDX_T> dst_selection;
  ifstream srcfile(argv[3]);
  string line;
  size_t lnum = 0;
  // reads node to rank assignment from file
  while (getline(srcfile, line))
    {
      istringstream iss(line);
      NODE_IDX_T n;
      
      assert (iss >> n);
      src_selection.insert(n);
      lnum++;
    }
  
  srcfile.close();

  ifstream dstfile(argv[4]);

  // reads node to rank assignment from file
  while (getline(dstfile, line))
    {
      istringstream iss(line);
      NODE_IDX_T n;
      
      assert (iss >> n);
      dst_selection.insert(n);
      lnum++;
    }
  dstfile.close();

  assert (!((src_selection.size() == 0) && (dst_selection.size() == 0)));
  // read the population info
  set< pair<pop_t, pop_t> > pop_pairs;
  assert(cell::read_population_combos(MPI_COMM_WORLD, input_file_name.c_str(), pop_pairs) >= 0);

  size_t total_num_nodes;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  assert(cell::read_population_ranges(MPI_COMM_WORLD, input_file_name.c_str(), pop_ranges, pop_vector, total_num_nodes) >= 0);

  vector< pair<string, string> > prj_names;
  assert(graph::read_projection_names(MPI_COMM_WORLD, input_file_name.c_str(), prj_names) >= 0);
      
  vector<NODE_IDX_T> edge_list;

  // read the edges
  for (size_t i = 0; i < prj_names.size(); i++)
    {
      size_t total_prj_num_edges = 0;
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base;
      NODE_IDX_T base, dst_start, src_start;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;

      if ((src_pop_name.compare(prj_names[i].first) == 0) &&
          (dst_pop_name.compare(prj_names[i].second) == 0))
        {
          printf("Reading projection %lu (%s -> %s)\n", i, prj_names[i].first.c_str(), prj_names[i].second.c_str());
          
          assert(graph::read_projection(MPI_COMM_WORLD, input_file_name, prj_names[i].first, prj_names[i].second, 
                                        dst_start, src_start, block_base, edge_base,
                                        dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                        total_prj_num_edges) >= 0);

          
          // validate the edges
          assert(graph::validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, pop_ranges, pop_pairs) == true);
          
          // filter/append to the edge list
          assert(filter_edge_list(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, src_selection, dst_selection, edge_list) >= 0);
        }
    }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (edge_list.size() > 0) 
    {
      ofstream outfile;
      stringstream outfilename;
      assert(edge_list.size()%2 == 0);

      outfilename << string(input_file_name) << "." << rank << ".subgraph";
      outfile.open(outfilename.str());

      for (size_t i = 0, k = 0; i < edge_list.size()-1; i+=2, k++)
        {
          outfile << k << " " << edge_list[i] << " " << edge_list[i+1] << std::endl;
        }
      outfile.close();

    }

  MPI_Finalize();
  return 0;
}
