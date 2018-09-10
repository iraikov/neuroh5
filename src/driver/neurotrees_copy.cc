// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_copy.cc
///
///  Driver program for copying tree structures.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
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

#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "rank_range.hh"
#include "read_tree.hh"
#include "validate_tree.hh"
#include "append_tree.hh"
#include "path_names.hh"
#include "create_file_toplevel.hh"
#include "exists_tree_h5types.hh"
#include "copy_tree_h5types.hh"


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
  printf("Usage: %s hdf-file population-name src-id dest-id...\n\n", argv[0]);
}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  int status;
  std::string pop_name;
  std::string input_filename, output_filename;
  CELL_IDX_T source_gid;
  std::vector<CELL_IDX_T> target_gid_list;
  vector<neurotree_t> input_tree_vec, output_tree_vec;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
  assert(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);

  int optflag_fill         = 0;
  int optflag_output       = 0;
  bool opt_attributes      = false;
  bool opt_fill            = false;
  bool opt_output_filename = false;
  // parse arguments
  static struct option long_options[] = {
    {"fill",    no_argument, &optflag_fill,  1 },
    {"output",  required_argument, &optflag_output,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hao:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          if (optflag_fill == 1) {
            opt_fill = true;
            optflag_fill = 0;
          }
          if (optflag_output == 1) {
            opt_output_filename = true;
            output_filename = string(optarg);
            optflag_output = 0;
          }
          break;
        case 'a':
          opt_attributes = true;
          break;
        case 'o':
          opt_output_filename = true;
          output_filename = string(optarg);
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if (optind < argc-2)
    {
      input_filename = std::string(argv[optind]);
      pop_name = std::string(argv[optind+1]);
      if (!opt_output_filename)
        {
          output_filename = input_filename;
        }
      {
        stringstream ss;
        ss << string(argv[optind+2]);
        ss >> source_gid;
      }
      if (optind+3 < argc)
        {
          for (int i = optind+3; i<argc; i++)
            {
              stringstream ss; CELL_IDX_T tree_id=0;
              ss << string(string(argv[i]));
              ss >> tree_id;
              target_gid_list.push_back(tree_id);
            }
        }
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  printf("Task %d: Population name is %s\n", rank, pop_name.c_str());
  printf("Task %d: Input file name is %s\n", rank, input_filename.c_str());
  printf("Task %d: Output file name is %s\n", rank, output_filename.c_str());
  printf("Task %d: Source id is %u\n", rank, source_gid);

  map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
  vector<pop_range_t> pop_vector;
  size_t n_nodes;
  
  // Read population info
  assert(cell::read_population_ranges(all_comm, input_filename,
                                      pop_ranges, pop_vector,
                                      n_nodes) >= 0);

  vector<pair <pop_t, string> > pop_labels;
  status = cell::read_population_labels(all_comm, input_filename, pop_labels);
  assert (status >= 0);
  
  // Determine index of population to be read
  size_t pop_idx=0; bool pop_idx_set=false;
  for (size_t i=0; i<pop_labels.size(); i++)
    {
      if (get<1>(pop_labels[i]) == pop_name)
        {
          pop_idx = get<0>(pop_labels[i]);
          pop_idx_set = true;
        }
    }
  if (!pop_idx_set)
    {
      throw_err("Population not found");
    }

  CELL_IDX_T pop_start = pop_vector[pop_idx].start;
  
  status = cell::read_trees (all_comm, input_filename,
                             pop_name, pop_start,
                             input_tree_vec);
  assert(status >= 0);
  assert(input_tree_vec.size() > 0);
  
  for_each(input_tree_vec.cbegin(),
           input_tree_vec.cend(),
           [&] (const neurotree_t& tree)
           { cell::validate_tree(tree); } 
           );

  const neurotree_t& input_tree = input_tree_vec.at(source_gid-pop_start);
      
  const vector<SECTION_IDX_T> & src_vector=get<1>(input_tree);
  const vector<SECTION_IDX_T> & dst_vector=get<2>(input_tree);
  const vector<SECTION_IDX_T> & sections=get<3>(input_tree);
  const vector<COORD_T> & xcoords=get<4>(input_tree);
  const vector<COORD_T> & ycoords=get<5>(input_tree);
  const vector<COORD_T> & zcoords=get<6>(input_tree);
  const vector<REALVAL_T> & radiuses=get<7>(input_tree);
  const vector<LAYER_IDX_T> & layers=get<8>(input_tree);
  const vector<PARENT_NODE_IDX_T> & parents=get<9>(input_tree);
  const vector<SWC_TYPE_T> & swc_types=get<10>(input_tree);
  

  if (opt_fill)
    {
      bool include_source_gid=false;
      if (input_filename.compare(output_filename) != 0)
        {
          include_source_gid=true;
        }
        
      target_gid_list.clear();
      for (size_t i=0; i<pop_vector[pop_idx].count; i++)
        {
          if (i != source_gid-pop_start)
            {
              target_gid_list.push_back(i);
            }
          else
            {
              if (include_source_gid)
                {
                  target_gid_list.push_back(i);
                }
            }
        }
    }
  
  // determine which trees are written by which rank
  vector< pair<hsize_t,hsize_t> > ranges;
  mpi::rank_ranges(target_gid_list.size(), size, ranges);

  hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;

  for (size_t i=start, ii=0; i<end; i++, ii++)
    {
      CELL_IDX_T gid = target_gid_list[i];
      printf("Task %d: Output id %u\n", rank, gid);

      neurotree_t tree = make_tuple(gid,
                                    src_vector, dst_vector, sections,
                                    xcoords, ycoords, zcoords,
                                    radiuses, layers, parents,
                                    swc_types);
      output_tree_vec.push_back(tree);

    }

  
  if (access( output_filename.c_str(), F_OK ) != 0)
    {
      vector <string> groups;
      groups.push_back (hdf5::POPULATIONS);
      status = hdf5::create_file_toplevel (all_comm, output_filename, groups);
    }
  assert(status == 0);

  // TODO; create separate functions for opening HDF5 file for reading and writing
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  assert(H5Pset_fapl_mpio(fapl, all_comm, MPI_INFO_NULL) >= 0);
  hid_t output_file = H5Fopen(output_filename.c_str(), H5F_ACC_RDWR, fapl);
  assert(output_file >= 0);
      
  if (!hdf5::exists_tree_h5types(output_file))
    {
      hid_t input_file = H5Fopen(input_filename.c_str(), H5F_ACC_RDONLY, fapl);
      assert(input_file >= 0);
      status = hdf5::copy_tree_h5types(input_file, output_file);
      status = H5Fclose (input_file);
      assert(status == 0);
    }

  assert(status == 0);
  status = H5Pclose (fapl);
  assert(status == 0);
  status = H5Fclose (output_file);
  assert(status == 0);

  MPI_Barrier(all_comm);

  status = cell::append_trees(all_comm, output_filename, pop_name, pop_start, output_tree_vec);
  assert(status == 0);

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
