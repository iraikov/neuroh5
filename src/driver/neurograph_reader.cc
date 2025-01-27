// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurograph_reader.cc
///
///  Driver program for read_graph function.
///
///  Copyright (C) 2016-2024 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_graph.hh"
#include "projection_names.hh"
#include "throw_assert.hh"

#include <mpi.h>

#include <getopt.h>

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace neuroh5;

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
                       edge_map_t& projection)
{
  DEBUG("output_projection: outfilename is ",outfilename,"\n");


  ofstream outfile;
  outfile.open(outfilename.c_str());

  for (edge_map_iter_t iter = projection.begin(); iter != projection.end(); iter++)
    {
      const NODE_IDX_T& dst = iter->first;
      const edge_tuple_t& tup = iter->second;
      
      const vector<NODE_IDX_T>& src_vector = get<0>(tup);
      const vector <data::AttrVal>&  edge_attr_val  = get<1>(tup);

      size_t i = 0;
      for (auto src : src_vector)
        {
          outfile << " " << src << " " << dst;
      
          for (const data::AttrVal& edge_attr_values : edge_attr_val)
            {
              
              for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
                {
                  outfile << " " << setprecision(9) << edge_attr_values.at<float>(j,i);
                }
              for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
                {
                  outfile << " " << (uint32_t)edge_attr_values.at<uint8_t>(j,i);
                }
              for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
                {
                  outfile << " " << edge_attr_values.at<uint16_t>(j,i);
                }
              for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
                {
                  outfile << " " << edge_attr_values.at<uint32_t>(j,i);
                }
            }
          i++;
          outfile << endl;
        }
    }

  outfile.close();

}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  string input_file_name;
  vector< string > edge_attr_name_spaces;

  throw_assert(MPI_Init(&argc, &argv) >= 0,
               "neurograph_reader: error in MPI initialization");

  int rank, size;
  throw_assert(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS,
               "neurograph_reader: error in MPI_Comm_size");
         
  throw_assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
               "neurograph_reader: error in MPI_Comm_rank");

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
            optflag_summary = 0;
          }
          if (optflag_verbose == 1) {
            debug_enabled = true;
            optflag_verbose = 0;
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
      input_file_name = string(argv[optind]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  vector< pair<string, string> > prj_names;
  throw_assert(graph::read_projection_names(MPI_COMM_WORLD, input_file_name,
                                            prj_names) >= 0,
               "neurograph_reader: error in reading projection names");


  vector<edge_map_t> prj_vector;
  size_t total_num_edges = 0, local_num_edges = 0, total_num_nodes = 0;

  vector < map <string, vector < vector<string> > > >  edge_attr_names_vector;
  
  // read the edges
  graph::read_graph (MPI_COMM_WORLD,
                     input_file_name,
                     edge_attr_name_spaces,
                     prj_names,
                     prj_vector,
                     edge_attr_names_vector,
                     total_num_nodes,
                     local_num_edges,
                     total_num_edges);

  printf("Task %d has read a total of %lu projections\n", rank,
         prj_vector.size());
  printf("Task %d has read a total of %lu edges\n", rank,  local_num_edges);
  printf("Task %d: total number of edges is %lu\n", rank,  total_num_edges);

  if (!opt_summary)
    {
      if (prj_vector.size() > 0)
        {
          for (size_t i = 0; i < prj_vector.size(); i++)
            {
              stringstream outfilename;
              outfilename << string(input_file_name) << "." << i << "." << rank
                          << ".edges";
              output_projection(outfilename.str(), prj_vector[i]);
            }
        }
    }

  MPI_Finalize();
  return 0;
}
