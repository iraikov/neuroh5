// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_import.cc
///
///  Driver program for various import procedures.
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
#include "read_layer_swc.hh"
#include "rank_range.hh"
#include "validate_tree.hh"
#include "append_tree.hh"
#include "insert_tree_points.hh"
#include "path_names.hh"
#include "create_file_toplevel.hh"


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
  cout << "Usage: " << string(argv[0]) << " [OPTIONS] <POPULATION> <HDF FILE> <SWC FILE>..." << endl <<
    "Options:" << endl <<
    "-h         Print this help" << endl <<
    "-s         Split morphology by layer" << endl <<
    "-d FILE    Read a list of ids and swc files from the given file " << endl <<
    "-l FILE    Read a list of SWC files from the given file and assign them sequential ids " << endl <<
    "-r FILE    Read singleton SWC file and assign it id 0 " << endl <<
    "-n OFFSET  Specify SWC node id offset " << endl <<
    "-o OFFSET  Specify cell id offset " << endl <<
    "-t SWCTYPE Specify default SWC type (indicates that input SWC has layer info instead of type) " << endl <<
    "-y OFFSET  Specify layer offset " << endl <<
    "-i FILE    Read given SWC file and prepend its points into every read file " << endl <<
    endl;
}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  int status;
  std::string pop_name;
  std::string output_file_name;
  std::string filelist_name, idfilelist_name, singleton_filename, include_filename;
  std::vector<std::string> input_file_names;
  std::vector<CELL_IDX_T> gid_list;
  int tree_id_offset=0, node_id_offset=0, layer_offset=0; int swc_type=0; int include_layer=0;
  vector<neurotree_t> tree_list, include_tree_list;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS);
  assert(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS);

  bool opt_include_layer  = false;
  bool opt_split_layers   = false;
  bool opt_layer_offset   = false;
  bool opt_node_id_offset = false;
  bool opt_tree_id_offset = false;
  bool opt_swctype        = false;
  bool opt_filelist       = false;
  bool opt_idfilelist     = false;
  bool opt_singleton      = false;
  bool opt_include        = false;
  // parse arguments
  static struct option long_options[] = {
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hd:e:i:o:r:t:l:n:sy:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          break;
        case 's':
          opt_split_layers = true;
          break;
        case 'd':
          opt_idfilelist = true;
          idfilelist_name = string(optarg);
          break;
        case 'i':
          opt_include = true;
          include_filename = string(optarg);
          break;
        case 'e':
          opt_include_layer = true;
          ss << string(optarg);
          ss >> include_layer;
          break;
        case 'r':
          opt_singleton = true;
          singleton_filename = string(optarg);
          break;
        case 'l':
          opt_filelist = true;
          filelist_name = string(optarg);
          break;
        case 'n':
          opt_node_id_offset = true;
          ss << string(optarg);
          ss >> node_id_offset;
          break;
        case 'o':
          opt_tree_id_offset = true;
          ss << string(optarg);
          ss >> tree_id_offset;
          break;
        case 't':
          opt_swctype = true;
          ss << string(optarg);
          ss >> swc_type;
          break;
        case 'y':
          opt_layer_offset = true;
          ss << string(optarg);
          ss >> layer_offset;
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
      pop_name = std::string(argv[optind]);
      output_file_name = std::string(argv[optind+1]);
      if (opt_singleton)
        {
          gid_list.push_back(0);
          input_file_names.push_back (singleton_filename);
        }
      else if (opt_idfilelist)
        {
          ifstream infile(idfilelist_name);
          string line;
          
          while (getline(infile, line))
            {
              stringstream ss;
              CELL_IDX_T gid; string filename;
              ss << line;
              ss >> gid;
              ss >> filename;
              input_file_names.push_back(filename);
              gid_list.push_back(gid);
            }
        }
      else if (opt_filelist)
        {
          ifstream infile(filelist_name);
          string line;
          
          CELL_IDX_T tree_id = tree_id_offset;
          while (getline(infile, line))
            {
              stringstream ss;
              string filename;
              ss << line;
              ss >> filename;
              input_file_names.push_back(filename);
              gid_list.push_back(tree_id);
              tree_id = tree_id+1;
            }
        }
      else
        {
          CELL_IDX_T tree_id = tree_id_offset;
          for (int i = optind+2; i<argc; i++)
            {
              input_file_names.push_back(std::string(argv[i]));
              gid_list.push_back(tree_id);
              tree_id = tree_id+1;
            }
        }
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  printf("Task %d: Population name is %s\n", rank, pop_name.c_str());
  printf("Task %d: Output file name is %s\n", rank, output_file_name.c_str());

  // determine which trees are read by which rank
  vector< pair<hsize_t,hsize_t> > ranges;
  mpi::rank_ranges(input_file_names.size(), size, ranges);

  size_t filecount=0;
  hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;

  assert(gid_list.size() > 0);

  
  if (opt_include)
    { 
      CELL_IDX_T gid0 = 0;
      // if swc type is not given, then we are reading a regular swc file
      status = io::read_swc (include_filename, gid0, 0, include_tree_list);
      cell::validate_tree(include_tree_list[0]);
    }

  if (opt_singleton)
    { // reading a single swc file with multiple ids
      std::string input_file_name = input_file_names[0];
      CELL_IDX_T gid0 = gid_list[0];
      if (opt_swctype) 
        {
          // if swc type is given, then we are reading an swc file with layer encoding
          status = io::read_layer_swc (input_file_name, gid0, node_id_offset, layer_offset,
                                       swc_type, opt_split_layers, tree_list);
        }
      else
        {
          // if swc type is not given, then we are reading a regular swc file
          status = io::read_swc (input_file_name, gid0, node_id_offset, tree_list);
        }
    }
  else
    {
      if (opt_swctype) 
        {
          // if swc type is given, then we are reading an swc file with layer encoding
          for (size_t i=start; i<end; i++)
            {
              std::string input_file_name = input_file_names[i];
              CELL_IDX_T gid = gid_list[i];
              status = io::read_layer_swc (input_file_name, gid, node_id_offset, layer_offset,
                                           swc_type, opt_split_layers, tree_list);
              filecount++;
              if (filecount % 1000 == 0)
                {
                  printf("Task %d: %lu trees read\n", rank,  filecount);
                }
            }
        }
      else
        {
          // if swc type is not given, then we are reading a regular swc file
          for (size_t i=start; i<end; i++)
            {
              std::string input_file_name = input_file_names[i];
              CELL_IDX_T gid = gid_list[i];
              status = io::read_swc (input_file_name, gid, node_id_offset, tree_list);
              filecount++;
              if (filecount % 1000 == 0)
                {
                  printf("Task %d: %lu trees read\n", rank,  filecount);
                }
            }
          
        }
    }
  
  printf("Task %d has read a total of %lu trees\n", rank,  tree_list.size());

  
  
  if (opt_include)
    {
      for (auto & tree : tree_list)
        {
          cell::insert_tree_points(include_tree_list[0], tree, include_layer);
        }
    }

  for (auto & tree : tree_list)
    {
      cell::validate_tree(tree);
    }
  
  if (access( output_file_name.c_str(), F_OK ) != 0)
    {
      vector <string> groups;
      groups.push_back (hdf5::POPULATIONS);
      status = hdf5::create_file_toplevel (all_comm, output_file_name, groups);
    }
  assert(status == 0);
  MPI_Barrier(all_comm);

  status = cell::append_trees(all_comm, output_file_name, pop_name, 0, tree_list);
  assert(status == 0);

  MPI_Barrier(all_comm);
  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
