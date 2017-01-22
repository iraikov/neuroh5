// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_import.cc
///
///  Driver program for various import procedures.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================


#include "debug.hh"

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

#include "neurotrees_types.hh"
#include "read_layer_swc.hh"
#include "dataset_num_elements.hh"
#include "rank_range.hh"
#include "write_tree.hh"
#include "hdf5_types.hh"
#include "hdf5_path_names.hh"
#include "hdf5_create_tree_file.hh"
#include "hdf5_create_tree_dataset.hh"
#include "hdf5_exists_tree_dataset.hh"


using namespace std;
using namespace neurotrees;


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
  printf("Usage: %s population-name hdf-file swc-file...\n\n", argv[0]);
}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  int status;
  std::string projection_name;
  std::string output_file_name;
  std::string hdf5_input_filename, hdf5_input_attrpath;
  std::vector<TREE_IDX_T> gid_list;
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
          opt_hdf5 = true;
          string arg = string(optarg);
          string delimiter = ":";
          size_t pos = arg.find(delimiter);
          hdf5_input_filename = arg.substr(0, pos); 
          hdf5_input_dsetpath = arg.substr(pos + delimiter.length(),
                                           arg.find(delimiter, pos + delimiter.length())); 
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

  // Populations/GC/Connectivity Group
  // /Populations/GC/Connectivity/source_gid Group
  // /Populations/GC/Connectivity/source_gid/gid Dataset {1000000/Inf}
  // /Populations/GC/Connectivity/source_gid/ptr Dataset {1000001/Inf}
  // /Populations/GC/Connectivity/source_gid/value Dataset {2399360031/Inf}
  // /Populations/GC/Connectivity/syn_id Group
  // /Populations/GC/Connectivity/syn_id/gid Dataset {1000000/Inf}
  // /Populations/GC/Connectivity/syn_id/ptr Dataset {1000001/Inf}
  // /Populations/GC/Connectivity/syn_id/value Dataset {2399360031/Inf}
  


  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
