// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file parts.cc
///
///  Driver program for graph partitioning functions.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "read_graph.hh"
#include "graph_parts.hh"
#include "model_types.hh"
#include "projection_names.hh"

#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include <mpi.h>
#include <parmetis.h>

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
  printf("Usage: %s [graphfile] [nparts] [options]\n\n", argv[0]);
  printf("Options:\n");
}




/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  std::string input_file_name, output;
  size_t nparts = 0, iosize = 0;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  debug_enabled = false;
  
  // parse arguments
  int optflag_nparts = 0;
  int optflag_iosize = 0;
  int optflag_output = 0;
  bool opt_nparts = false,
    opt_iosize = false,
    opt_output = false;

  static struct option long_options[] = {
    {"output",    required_argument, &optflag_output,  1 },
    {"nparts",    required_argument, &optflag_nparts,  1 },
    {"iosize",    required_argument, &optflag_iosize,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hi:n:o:",
			   long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          if (optflag_nparts == 1) {
            opt_nparts = true;
            ss << string(optarg);
            ss >> nparts;
          }
          if (optflag_iosize == 1) {
            opt_iosize = true;
            ss << string(optarg);
            ss >> iosize;
          }
          if (optflag_output == 1) {
            opt_output = true;
            output = string(optarg);
          }
          break;
        case 'o':
          opt_output = true;
          output = string(optarg);
          break;
        case 'n':
          opt_nparts = true;
          ss << string(optarg);
          ss >> nparts;
          break;
        case 'i':
          opt_iosize = true;
          ss << string(optarg);
          ss >> iosize;
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  printf("nparts = %lu\n", nparts);
  printf("iosize = %lu\n", iosize);

  if (opt_nparts && (optind < argc))
    {
      input_file_name = std::string(argv[optind]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  if (!opt_iosize) iosize = 4;

  vector<string> prj_names;
  assert(io::hdf5::read_projection_names(MPI_COMM_WORLD, input_file_name, prj_names) >= 0);

  vector<model::prj_tuple_t> prj_list;
  size_t total_num_edges = 0, local_num_edges = 0, total_num_nodes = 0;
  
  // read the edges
  std::vector<idx_t> parts;
  
  graph::partition_graph
  (
   MPI_COMM_WORLD,
   input_file_name,
   prj_names,
   iosize,
   nparts,
   parts
   );

  if (!opt_output)
    {
      for (size_t i = 0; i < parts.size(); i++)
        {
          cout << parts[i] << std::endl;
        }
    }
  else
    {
      ofstream outfile;
      stringstream outfilename;
      outfilename << output << "." << nparts << "." << rank;
      outfile.open(outfilename.str().c_str());
      for (size_t i = 0; i < parts.size(); i++)
        {
          outfile << parts[i] << std::endl;
        }
      outfile.flush();
      outfile.close();
    }

  MPI_Finalize();
  return 0;
}
