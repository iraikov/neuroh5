#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"

#include "hdf5.h"

#include <getopt.h>
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

void throw_err(char const* err_message)
{
  fprintf(stderr, "Error: %s\n", err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task)
{
  fprintf(stderr, "Task %d Error: %s\n", task, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task, int32_t thread)
{
  fprintf(stderr, "Task %d Thread %d Error: %s\n", task, thread, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}


void print_usage_full(char** argv)
{
  printf("Usage: %s [graphfile] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-s:\n");
  printf("\t\tPrint only edge summary\n");
}

/*****************************************************************************
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/

int append_edge_list
(
 const NODE_IDX_T&         base,
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 size_t&                   num_edges,
 vector<NODE_IDX_T>&       edge_list
 )
{
  int ierr = 0; size_t dst_ptr_size;
  num_edges = 0;
  
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
                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  for (size_t j = low; j < high; ++j)
                    {
                      NODE_IDX_T src = src_idx[j] + src_start;
                      edge_list.push_back(src);
                      edge_list.push_back(dst);
		      num_edges++;
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
  
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments
  int optflag_summary = 0;
  bool opt_summary = false;
  static struct option long_options[] = {
    {"summary",    no_argument, &optflag_summary,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "sh",
			   long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_summary == 1) {
            opt_summary = true;
          }
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        case 's':
          opt_summary = true;
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if (optind < argc)
    {
      input_file_name = argv[optind];
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }
 
  // read the population info
  set< pair<pop_t, pop_t> > pop_pairs;
  assert(read_population_combos(MPI_COMM_WORLD, input_file_name, pop_pairs) >= 0);

  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  assert(read_population_ranges(MPI_COMM_WORLD, input_file_name, pop_ranges, pop_vector) >= 0);

  vector<string> prj_names;
  assert(read_projection_names(MPI_COMM_WORLD, input_file_name, prj_names) >= 0);
      
  vector<NODE_IDX_T> edge_list;

  size_t total_num_edges = 0, local_num_edges = 0;
  
  // read the edges
  for (size_t i = 0; i < prj_names.size(); i++)
    {
      NODE_IDX_T base, dst_start, src_start;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;
      size_t local_prj_num_edges = 0, total_prj_num_edges = 0;
      
      printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

      assert(read_dbs_projection(MPI_COMM_WORLD, input_file_name, prj_names[i].c_str(), 
                                 pop_vector, total_prj_num_edges, base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
      // validate the edges
      assert(validate_edge_list(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, pop_ranges, pop_pairs) == true);
      
      // append to the partitioner input list
      assert(append_edge_list(base, dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, local_prj_num_edges, edge_list) >= 0);

      // ensure that all edges in the projection have been read and appended to edge_list
      assert(local_prj_num_edges == src_idx.size());

      printf("Task %d has read %lu edges in projection %lu (%s)\n", rank,  local_prj_num_edges, i, prj_names[i].c_str());

      total_num_edges = total_num_edges + total_prj_num_edges;
      local_num_edges = local_num_edges + local_prj_num_edges;
    }

  assert(local_num_edges == edge_list.size()/2);

  printf("Task %d has read a total of %lu edges\n", rank,  local_num_edges);
  printf("Task %d: total number of edges is %lu\n", rank,  total_num_edges);
  
  size_t sum_local_num_edges = 0;
  MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
	     MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
    {
      assert(sum_local_num_edges == total_num_edges);
    }
  
  if (!opt_summary)
    {
      if (edge_list.size() > 0) 
        {
          ofstream outfile;
          stringstream outfilename;
          assert(edge_list.size()%2 == 0);
          
          outfilename << string(input_file_name) << "." << rank << ".edges";
          outfile.open(outfilename.str());
          
          for (size_t i = 0, k = 0; i < edge_list.size()-1; i+=2, k++)
            {
              outfile << k << " " << edge_list[i] << " " << edge_list[i+1] << std::endl;
            }
          outfile.close();
        }
    }

  MPI_Finalize();
  return 0;
}
