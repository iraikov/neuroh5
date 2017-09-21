// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_read.cc
///
///  Driver program for read_trees function.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "read_tree.hh"
#include "validate_tree.hh"
#include "path_names.hh"
#include "dataset_num_elements.hh"

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
  printf("Usage: %s [treefile] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t--verbose:\n");
  printf("\t\tPrint verbose diagnostic information\n");
}


/*****************************************************************************
 * Prints out tree content
 *****************************************************************************/

void output_tree(string outfilename,
                 const neurotree_t& tree)
{
  DEBUG("output_tree: outfilename is ",outfilename,"\n");

  ofstream outfile;
  outfile.open(outfilename.c_str());

  const std::vector<SECTION_IDX_T> & src_vector=get<1>(tree);
  const std::vector<SECTION_IDX_T> & dst_vector=get<2>(tree);
  const std::vector<SECTION_IDX_T> & sections=get<3>(tree);
  const std::vector<COORD_T> & xcoords=get<4>(tree);
  const std::vector<COORD_T> & ycoords=get<5>(tree);
  const std::vector<COORD_T> & zcoords=get<6>(tree);
  /*const std::vector<REALVAL_T> & radiuses=get<7>(tree);*/
  const std::vector<LAYER_IDX_T> & layers=get<8>(tree);
  const std::vector<PARENT_NODE_IDX_T> & parents=get<9>(tree);
  const std::vector<SWC_TYPE_T> & swc_types=get<10>(tree);
  
  outfile << "number of x points: " << xcoords.size() << endl;
  outfile << "number of y points: " << ycoords.size() << endl;
  outfile << "number of z points: " << zcoords.size() << endl;
  
  outfile << "layers: " << endl;
  for_each(layers.cbegin(),
           layers.cend(),
           [&] (const LAYER_IDX_T i)
           { outfile << " " << i; } 
           );
  outfile << endl;

  outfile << "SWC types: " << endl;
  for_each(swc_types.cbegin(),
           swc_types.cend(),
           [&] (const SWC_TYPE_T i)
           { outfile << " " << (int)i; } 
           );
  outfile << endl;
  
  outfile << "src_vector: " << endl;
  for_each(src_vector.cbegin(),
           src_vector.cend(),
           [&] (const Graph::vertex i)
           { outfile << " " << i; } 
           );
  outfile << endl;
  
  outfile << "dst_vector: " << endl;
  for_each(dst_vector.cbegin(),
           dst_vector.cend(),
           [&] (const Graph::vertex i)
           { outfile << " " << i; } 
           );
  outfile << endl;
  
  outfile << "sec_vector: " << endl;
  for_each(sections.cbegin(),
           sections.cend(),
           [&] (const Graph::vertex i)
           { outfile << " " << i; } 
           );
  outfile << endl;

  outfile << "parents: " << endl;
  for_each(parents.cbegin(),
           parents.cend(),
           [&] (const int i)
           { outfile << " " << i; } 
           );
  outfile << endl;

  outfile.close();

}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  herr_t status;
  MPI_Comm all_comm;
  string input_file_name;

  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  debug_enabled = false;

  // parse arguments
  int optflag_verbose = 0;
  static struct option long_options[] = {
    {"verbose",  no_argument, &optflag_verbose,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "h",
                           long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_verbose == 1) {
            debug_enabled = true;
            optflag_verbose = 0;
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

  if (optind < argc)
    {
      input_file_name = string(argv[optind]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
  vector<pop_range_t> pop_vector;
  size_t n_nodes;
  // Read population info
  assert(cell::read_population_ranges(all_comm, input_file_name,
                                      pop_ranges, pop_vector,
                                      n_nodes) >= 0);

  vector<string> pop_names;
  status = cell::read_population_names(all_comm, input_file_name, pop_names);
  assert (status >= 0);

  size_t start=0, end=0;
  std::vector<neurotree_t> tree_list;
  for (size_t i = 0; i<pop_names.size(); i++)
    {
      status = cell::read_trees (all_comm, input_file_name,
                                 pop_names[i], pop_vector[i].start,
                                 tree_list, start, end, true);
      assert (status >= 0);
      
      for_each(tree_list.cbegin(),
               tree_list.cend(),
               [&] (const neurotree_t& tree)
               { cell::validate_tree(tree); } 
               );

    }
  
  size_t local_num_trees = tree_list.size();
  
  printf("Task %d has read a total of %lu trees\n", rank,  local_num_trees);

  if (tree_list.size() > 0)
    {
      for (size_t i = 0; i < tree_list.size(); i++)
        {
          stringstream outfilename;
          outfilename << string(input_file_name) << "." << i << "." << rank
                      << ".trees";
          output_tree(outfilename.str(), tree_list[i]);
        }
    }

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  return 0;
}
