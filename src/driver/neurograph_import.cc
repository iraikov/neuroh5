// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurograph_import.cc
///
///  Driver program for various import procedures.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "graph_reader.hh"
#include "read_graph.hh"
#include "model_types.hh"
#include "population_reader.hh"
#include "projection_names.hh"
#include "read_syn_projection.hh"
#include "write_connectivity.hh"

#include <mpi.h>
#include <hdf5.h>
#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>


using namespace std;
using namespace ngh5;


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
  printf("Usage: %s population-name hdf-file ...\n\n", argv[0]);
}


int append_edge_list
(
 const NODE_IDX_T&          dst_start,
 const NODE_IDX_T&          src_start,
 const vector<NODE_IDX_T>&  dst_idx,
 const vector<DST_PTR_T>&   src_idx_ptr,
 const vector<NODE_IDX_T>&  src_idx,
 const vector<DST_PTR_T>&   syn_idx_ptr,
 const vector<NODE_IDX_T>&  syn_idx,
 size_t&                    num_edges,
 vector<NODE_IDX_T>&        edge_list
 EdgeNamedAttr&             edge_attr_values
 )
{
  int ierr = 0; size_t src_idx_ptr_size;
  num_edges = 0;

  if (dst_idx.size() > 0)
    {
      for (size_t d = 0; d < dst_idx.size()-1; d++)
        {
          NODE_IDX_T dst = dst_idx[d] + dst_start;
          
          size_t low_src_ptr = src_idx_ptr[d],
            high_src_ptr = src_idx_ptr[d+1];
          size_t low_syn_ptr = syn_idx_ptr[d],
            high_syn_ptr = syn_idx_ptr[d+1];

          for (size_t i = low_src_ptr, ii = low_syn_ptr; i < high_src_ptr; ++i, ++ii)
            {
              assert(ii < high_syn_ptr);
              NODE_IDX_T src = src_idx[i] + src_start;
              NODE_IDX_T syn_id = syn_idx[ii];
              edge_list.push_back(src);
              edge_list.push_back(dst);
              edge_attr_values.push_back<uint32_t>(0, syn_id);
              num_edges++;
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
  int status=0;
  std::string projection_name;
  std::string output_file_name;
  std::string hdf5_input_filename, hdf5_input_dsetpath;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  bool opt_hdf5 = false;
  // parse arguments
  static struct option long_options[] = {
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hf:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          break;
        case 'f':
          {
            opt_hdf5 = true;
            string arg = string(optarg);
            string delimiter = ":";
            size_t pos = arg.find(delimiter);
            hdf5_input_filename = arg.substr(0, pos); 
            hdf5_input_dsetpath = arg.substr(pos + delimiter.length(),
                                             arg.find(delimiter, pos + delimiter.length()));
          }
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if (optind < argc-1)
    {
      projection_name  = std::string(argv[optind]);
      output_file_name = std::string(argv[optind+1]);
      if (!opt_hdf5)
        {
          print_usage_full(argv);
          exit(1);
        }
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  printf("Task %d: Projection name is %s\n", rank, projection_name.c_str());
  printf("Task %d: Output file name is %s\n", rank, output_file_name.c_str());

  vector<NODE_IDX_T>  dst_idx;
  vector<DST_PTR_T>   src_idx_ptr;
  vector<NODE_IDX_T>  src_idx;
  vector<DST_PTR_T>   syn_idx_ptr;
  vector<NODE_IDX_T>  syn_idx;

  status = ngh5::io::hdf5::read_syn_projection (all_comm,
                                                hdf5_input_filename,
                                                hdf5_input_dsetpath,
                                                dst_idx,
                                                src_idx_ptr,
                                                src_idx,
                                                syn_idx_ptr,
                                                syn_idx);
  vector<NODE_IDX_T>  edges;
  size_t num_edges;
  
  status = append_edge_list (dst_start, src_start,
                             dst_idx,
                             src_idx_ptr, src_idx,
                             syn_idx_ptr, syn_idx,
                             num_edges,
                             edges);

  ngh5::io::hdf5::write_connectivity (file, dst_start, dst_end, edges);

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
