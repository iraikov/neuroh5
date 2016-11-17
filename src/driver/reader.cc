// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file reader.cc
///
///  Driver program for read_graph function.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================


#include "debug.hh"
#include "ngh5_types.hh"
#include "model_types.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "projection_names.hh"
#include "graph_reader.hh"

#include "hdf5.h"

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

using namespace std;
using namespace ngh5;
using namespace ngh5::model;

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
  printf("\t-a:\n");
  printf("\t\tInclude edge attribute information\n");
  printf("\t-s:\n");
  printf("\t\tPrint only edge summary\n");
  printf("\t--verbose:\n");
  printf("\t\tPrint verbose diagnostic information\n");
}


/*****************************************************************************
 * Prints out projection content
 *****************************************************************************/

void output_projection(string outfilename,
                       const prj_tuple_t& projection)
{
  DEBUG("output_projection: outfilename is ",outfilename,"\n");
  
  const vector<NODE_IDX_T>& src_list = get<0>(projection);
  const vector<NODE_IDX_T>& dst_list = get<1>(projection);
  const EdgeAttr&  edge_attr_values  = get<2>(projection);

  ofstream outfile;
  outfile.open(outfilename);

  for (size_t i = 0; i < src_list.size(); i++)
    {
      outfile << i << " " << src_list[i] << " " << dst_list[i];
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
        {
          outfile << " " << setprecision(9) << edge_attr_values.at<float>(j,i); 
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
        {
          outfile << " " << edge_attr_values.at<uint8_t>(j,i); 
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
        {
          outfile << " " << edge_attr_values.at<uint16_t>(j,i);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
        {
          outfile << " " << edge_attr_values.at<uint32_t>(j,i);
        }

      outfile << std::endl;
    }

  outfile.close();

}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  std::string input_file_name;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  debug_enabled = false;
  
  // parse arguments
  int optflag_summary = 0;
  int optflag_verbose = 0;
  bool opt_summary = false;
  bool opt_attrs = false;
  static struct option long_options[] = {
    {"summary",  no_argument, &optflag_summary,  1 },
    {"verbose",  no_argument, &optflag_verbose,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "ahs",
			   long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_summary == 1) {
            opt_summary = true;
          }
          if (optflag_verbose == 1) {
            debug_enabled = true;
          }
          break;
        case 'a':
          opt_attrs = true;
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
      input_file_name = std::string(argv[optind]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }
 

  vector<string> prj_names;
  assert(io::hdf5::read_projection_names(MPI_COMM_WORLD, input_file_name,
                                         prj_names) >= 0);

  vector<prj_tuple_t> prj_list;
  size_t total_num_edges = 0, local_num_edges = 0, total_num_nodes = 0;
  
  // read the edges
  read_graph (MPI_COMM_WORLD,
              input_file_name,
              opt_attrs,
              prj_names,
              prj_list,
              total_num_nodes,
              local_num_edges,
              total_num_edges);

  
  printf("Task %d has read a total of %lu projections\n", rank,  prj_list.size());
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
      if (prj_list.size() > 0) 
        {
          for (size_t i = 0; i < prj_list.size(); i++)
            {
              stringstream outfilename;
              outfilename << string(input_file_name) << "." << i << "." << rank << ".edges";
              output_projection(outfilename.str(), prj_list[i]);
            }
        }
    }

  MPI_Finalize();
  return 0;
}
