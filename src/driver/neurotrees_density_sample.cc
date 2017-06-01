// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file density_sample.cc
///
///  Driver program for density_sample function.
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
using namespace neuroio;


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
  std::string pop_name;
  std::string output_file_name;
  std::string filelist_name, idfilelist_name;
  std::vector<std::string> input_file_names;
  std::vector<CELL_IDX_T> gid_list;
  int tree_id_offset=0, node_id_offset; int swc_type=0;
  vector<neurotree_t> tree_list;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  bool opt_node_id_offset = false;
  bool opt_tree_id_offset = false;
  bool opt_swctype = false;
  bool opt_filelist = false;
  bool opt_idfilelist = false;
  // parse arguments
  static struct option long_options[] = {
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hd:o:t:l:n:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          break;
        case 'd':
          opt_idfilelist = true;
          idfilelist_name = string(optarg);
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
      if (opt_idfilelist)
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
          for (size_t i = optind+2; i<argc; i++)
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

  // determine which blocks of block_ptr are read by which rank
  vector< pair<hsize_t,hsize_t> > ranges;
  rank_ranges(input_file_names.size(), size, ranges);

  size_t filecount=0;
  hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;
  
  for (size_t i=start; i<end; i++)
    {
      std::string input_file_name = input_file_names[i];
      CELL_IDX_T gid = gid_list[i];
      status = read_layer_swc (input_file_name, gid, node_id_offset, swc_type, tree_list);
      filecount++;
      if (filecount % 1000 == 0)
        {
          printf("Task %d: %lu trees read\n", rank,  filecount);
        }
    }
  
  printf("Task %d has read a total of %lu trees\n", rank,  tree_list.size());

  if (access( output_file_name.c_str(), F_OK ) != 0)
    {
      status = hdf5_create_tree_file (all_comm, output_file_name);
    }
  assert(status == 0);
  MPI_Barrier(all_comm);
  
  // TODO; create separate functions for opening HDF5 file for reading and writing
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  assert(H5Pset_fapl_mpio(fapl, all_comm, MPI_INFO_NULL) >= 0);
  hid_t file = H5Fopen(output_file_name.c_str(), H5F_ACC_RDWR, fapl);
  assert(file >= 0);

  if (!hdf5_exists_tree_dataset(file, pop_name))
    {
      status = hdf5_create_tree_dataset(all_comm, file, pop_name);
    }
  assert(status == 0);

  // Reads the current attribute extent to determine number of entries for that population
  hsize_t ptr_num   = dataset_num_elements(all_comm, file, tree_attribute_path(TREES, pop_name, ATTR_PTR));
  hsize_t attr_num  = dataset_num_elements(all_comm, file, tree_attribute_path(TREES, pop_name, X_COORD));
  hsize_t sec_num   = dataset_num_elements(all_comm, file, tree_attribute_path(TREES, pop_name, SECTION));
  hsize_t topo_num  = dataset_num_elements(all_comm, file, tree_attribute_path(TREES, pop_name, SRCSEC));
  hsize_t ptr_start = 0, attr_start = 0, sec_start = 0, topo_start = 0;
  status = H5Pclose (fapl);
  status = H5Fclose (file);
  
  if (ptr_num > 0)  ptr_start  = ptr_num-1;
  attr_start = attr_num;
  sec_start  = sec_num;
  topo_start = topo_num;
  
  status = write_trees(all_comm, output_file_name, pop_name, 
                       ptr_start, attr_start, sec_start, topo_start, 
                       tree_list);
  assert(status == 0);


  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
