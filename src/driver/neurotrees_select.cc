// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_select.cc
///
///  Program for selecting tree subsets.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================


#include "debug.hh"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <getopt.h>
#include <mpi.h>

#include "neuroio_types.hh"
#include "hdf5_path_names.hh"
#include "scatter_read_tree.hh"
#include "write_tree.hh"
#include "cell_attributes.hh"
#include "read_population_names.hh"
#include "read_population_ranges.hh"
#include "dataset_num_elements.hh"
#include "validate_tree.hh"
#include "create_file_toplevel.hh"
#include "create_tree_dataset.hh"
#include "exists_tree_dataset.hh"
#include "exists_tree_h5types.hh"
#include "copy_tree_h5types.hh"

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
  printf("Usage: %s [treefile] [options] INPUT_FILE SELECTION_FILE OUTPUT_FILE\n\n", argv[0]);
  printf("Options:\n");
  printf("\t--verbose:\n");
  printf("\t\tPrint verbose diagnostic information\n");
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  herr_t status;
  MPI_Comm all_comm;
  string pop_name, input_file_name, output_file_name, selection_file_name, rank_file_name, attr_namespace = "Attributes";
  size_t n_nodes;
  map<CELL_IDX_T, size_t> node_rank_map;
  stringstream ss;

  assert(MPI_Init(&argc, &argv) >= 0);

  size_t chunksize=1000, value_chunksize=1000, cachesize=1*1024*1024;
  int rank, size, io_size=1;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  debug_enabled = false;

  // parse arguments
  int optflag_verbose   = 0;
  int optflag_rankfile  = 0;
  int optflag_iosize    = 0;
  int optflag_namespace = 0;
  int optflag_chunksize = 0;
  int optflag_value_chunksize = 0;
  int optflag_cachesize = 0;
  bool opt_rankfile = false,
    opt_iosize      = false,
    opt_attrs       = false,
    opt_namespace   = false,
    opt_population  = false,
    opt_chunksize   = false,
    opt_value_chunksize  = false,
    opt_cachesize   = false;
  
  static struct option long_options[] = {
    {"verbose",   no_argument, &optflag_verbose,  1 },
    {"rankfile",  required_argument, &optflag_rankfile,  1 },
    {"iosize",    required_argument, &optflag_iosize,  1 },
    {"namespace", required_argument, &optflag_namespace,  1 },
    {"chunksize", required_argument, &optflag_chunksize,  1 },
    {"value-chunksize", required_argument, &optflag_value_chunksize,  1 },
    {"cachesize", required_argument, &optflag_cachesize,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "ai:p:r:n:oh",
                           long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_rankfile == 1) {
            opt_rankfile = true;
            rank_file_name = std::string(strdup(optarg));
          }
          if (optflag_namespace == 1) {
            opt_namespace = true;
            attr_namespace = std::string(strdup(optarg));
          }
          if (optflag_iosize == 1) {
            opt_iosize = true;
            ss << string(optarg);
            ss >> io_size;
          }
          if (optflag_chunksize == 1) {
            opt_chunksize = true;
            ss << string(optarg);
            ss >> chunksize;
          }
          if (optflag_value_chunksize == 1) {
            opt_value_chunksize = true;
            ss << string(optarg);
            ss >> value_chunksize;
          }
          if (optflag_cachesize == 1) {
            opt_cachesize = true;
            ss << string(optarg);
            ss >> cachesize;
          }
          if (optflag_verbose == 1) {
            debug_enabled = true;
          }
          break;
        case 'a':
          opt_attrs = true;
          break;
        case 'i':
          opt_iosize = true;
          ss << string(optarg);
          ss >> io_size;
          break;
        case 'n':
          opt_namespace = true;
          attr_namespace = std::string(strdup(optarg));
          break;
        case 'p':
          opt_population  = true;
          pop_name = std::string(strdup(optarg));
          break;
        case 'r':
          opt_rankfile = true;
          rank_file_name = std::string(strdup(optarg));
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if (optind <= argc-3)
    {
      input_file_name = string(argv[optind]);
      selection_file_name = string(argv[optind+1]);
      output_file_name = string(argv[optind+2]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
  vector<pop_range_t> pop_vector;

  // Read population info to determine n_nodes
  assert(read_population_ranges(all_comm, input_file_name,
                                pop_ranges, pop_vector,
                                n_nodes) >= 0);
  // TODO; create separate functions for opening HDF5 file for reading and writing
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  assert(H5Pset_fapl_mpio(fapl, all_comm, MPI_INFO_NULL) >= 0);
  hid_t input_file = H5Fopen(input_file_name.c_str(), H5F_ACC_RDONLY, fapl);
  assert(input_file >= 0);

  vector<string> pop_names;
  status = read_population_names(all_comm, input_file, pop_names);
  assert (status >= 0);

  // Determine index of population to be read
  size_t pop_idx;
  for (pop_idx=0; pop_idx<pop_names.size(); pop_idx++)
    {
      if (pop_names[pop_idx] == pop_name)
        break;
    }
  if (pop_idx >= pop_names.size())
    {
      throw_err("Population not found");
    }
  
  status = H5Pclose (fapl);
  status = H5Fclose (input_file);

  // Read in selection gids
  set<CELL_IDX_T> tree_selection;
  {
    ifstream infile(selection_file_name.c_str());
    string line;
    // reads gid per line
    while (getline(infile, line))
      {
        istringstream iss(line);
        CELL_IDX_T n;
        assert (iss >> n);
        tree_selection.insert(n);
      }
    
    infile.close();
  }

  // Determine which nodes are assigned to which compute ranks
  if (!opt_rankfile)
    {
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < n_nodes; i++)
        {
          node_rank_map.insert(make_pair(i,i%size));
        }
    }
  else
    {
      ifstream infile(rank_file_name.c_str());
      string line;
      size_t i = 0;
      // reads node to rank assignment from file
      while (getline(infile, line))
        {
          istringstream iss(line);
          size_t n;

          assert (iss >> n);
          node_rank_map.insert(make_pair(i,n));
          i++;
        }

      infile.close();
    }

  map<CELL_IDX_T, neurotree_t>  tree_map;
  NamedAttrMap attr_map;
  
  status = scatter_read_trees (all_comm, input_file_name, io_size,
                               opt_attrs, attr_namespace,
                               node_rank_map,
                               pop_name, pop_vector[pop_idx].start,
                               tree_map, attr_map);
  
  
  assert (status >= 0);
  
  for_each(tree_map.cbegin(),
           tree_map.cend(),
           [&] (const pair<CELL_IDX_T, neurotree_t> &element)
           { const neurotree_t& tree = element.second;
             validate_tree(tree); } 
           );
  
  size_t local_num_trees = tree_map.size();
  
  printf("Task %d has received a total of %lu trees\n", rank,  local_num_trees);
  vector <size_t> num_attrs;
  num_attrs.resize(AttrMap::num_attr_types);

  vector<vector<string>> attr_names;
  attr_names.resize(AttrMap::num_attr_types);

  attr_map.attr_names(attr_names);
  attr_map.num_attrs(num_attrs);


  vector<map< CELL_IDX_T, vector<float> > >    subset_float_values(num_attrs[AttrMap::attr_index_float]);
  vector<map< CELL_IDX_T, vector<uint8_t> > >  subset_uint8_values(num_attrs[AttrMap::attr_index_uint8]);
  vector<map< CELL_IDX_T, vector<int8_t> > >   subset_int8_values(num_attrs[AttrMap::attr_index_int8]);
  vector<map< CELL_IDX_T, vector<uint16_t> > > subset_uint16_values(num_attrs[AttrMap::attr_index_uint16]);
  vector<map< CELL_IDX_T, vector<uint32_t> > > subset_uint32_values(num_attrs[AttrMap::attr_index_uint32]);
  vector<map< CELL_IDX_T, vector<int32_t> > >  subset_int32_values(num_attrs[AttrMap::attr_index_int32]);

  vector<neurotree_t> tree_subset;
  
  for (auto const& element : tree_map)
    {
      const CELL_IDX_T gid = element.first;
      if (tree_selection.find(gid) != tree_selection.end())
        {
          const neurotree_t &tree = element.second;
          tree_subset.push_back(tree);

          const vector<vector<float>>    float_values  = attr_map.find<float>(gid);
          const vector<vector<uint8_t>>  uint8_values  = attr_map.find<uint8_t>(gid);
          const vector<vector<int8_t>>   int8_values   = attr_map.find<int8_t>(gid);
          const vector<vector<uint16_t>> uint16_values = attr_map.find<uint16_t>(gid);
          const vector<vector<uint32_t>> uint32_values = attr_map.find<uint32_t>(gid);
          const vector<vector<int32_t>>  int32_values  = attr_map.find<int32_t>(gid);

          for (size_t i=0; i<float_values.size(); i++)
            {
              subset_float_values[i].insert(make_pair(gid, float_values[i]));
            }
          for (size_t i=0; i<uint8_values.size(); i++)
            {
              subset_uint8_values[i].insert(make_pair(gid, uint8_values[i]));
            }
          for (size_t i=0; i<int8_values.size(); i++)
            {
              subset_int8_values[i].insert(make_pair(gid, int8_values[i]));
            }
          for (size_t i=0; i<uint16_values.size(); i++)
            {
              subset_uint16_values[i].insert(make_pair(gid, uint16_values[i]));
            }
          for (size_t i=0; i<uint32_values.size(); i++)
            {
              subset_uint32_values[i].insert(make_pair(gid, uint32_values[i]));
            }
          for (size_t i=0; i<int32_values.size(); i++)
            {
              subset_int32_values[i].insert(make_pair(gid, int32_values[i]));
            }
        }
    }

  tree_map.clear();
  
  // Determine the total number of selected trees
  uint32_t global_subset_size=0, local_subset_size=tree_subset.size();

  assert(MPI_Reduce(&local_subset_size, &global_subset_size, 1, MPI_UINT32_T,
                    MPI_SUM, 0, all_comm) >= 0);
  assert(MPI_Bcast(&global_subset_size, 1, MPI_UINT32_T, 0, all_comm) >= 0);

  printf("Task %d local selection size is %lu\n", rank, local_subset_size);

  hsize_t ptr_start = 0, attr_start = 0, sec_start = 0, topo_start = 0;

  if (global_subset_size > 0)
    {
      //status = access( output_file_name.c_str(), F_OK );
      status = hdf5_create_tree_file (all_comm, output_file_name);
      assert(status == 0);
      MPI_Barrier(all_comm);
      
      // TODO; create separate functions for opening HDF5 file for reading and writing
      fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, all_comm, MPI_INFO_NULL) >= 0);
      hid_t output_file = H5Fopen(output_file_name.c_str(), H5F_ACC_RDWR, fapl);
      assert(output_file >= 0);
      
      if (!hdf5_exists_tree_dataset(output_file, pop_name))
        {
          status = hdf5_create_tree_dataset(all_comm, output_file, pop_name);
        }

      if (!hdf5_exists_tree_h5types(output_file))
        {
          input_file = H5Fopen(input_file_name.c_str(), H5F_ACC_RDONLY, fapl);
          assert(input_file >= 0);
          status = hdf5_copy_tree_h5types(input_file, output_file);
          status = H5Fclose (input_file);
          assert(status == 0);
        }

      assert(status == 0);
      status = H5Pclose (fapl);
      assert(status == 0);
      status = H5Fclose (output_file);
      assert(status == 0);
      
      status = write_trees(all_comm, output_file_name, pop_name, 
                           ptr_start, attr_start, sec_start, topo_start, 
                           tree_subset);
      
      assert(status == 0);

      if (opt_attrs)
        {
          for (size_t i=0; i<subset_float_values.size(); i++)
            {
              write_cell_attribute_map<float>(all_comm,
                                              output_file_name,
                                              attr_namespace,
                                              pop_name,
                                              attr_names[AttrMap::attr_index_float][i],
                                              subset_float_values[i],
                                              chunksize,
                                              value_chunksize,
                                              cachesize);
            }
          for (size_t i=0; i<subset_uint8_values.size(); i++)
            {
              write_cell_attribute_map<uint8_t>(all_comm,
                                                output_file_name,
                                                attr_namespace,
                                                pop_name,
                                                attr_names[AttrMap::attr_index_uint8][i],
                                                subset_uint8_values[i],
                                                chunksize,
                                                value_chunksize,
                                                cachesize);
            }
          for (size_t i=0; i<subset_int8_values.size(); i++)
            {
              write_cell_attribute_map<int8_t>(all_comm,
                                               output_file_name,
                                               attr_namespace,
                                               pop_name,
                                               attr_names[AttrMap::attr_index_int8][i],
                                               subset_int8_values[i],
                                               chunksize,
                                               value_chunksize,
                                               cachesize);
            }
          for (size_t i=0; i<subset_uint16_values.size(); i++)
            {
              write_cell_attribute_map<uint16_t>(all_comm,
                                                 output_file_name,
                                                 attr_namespace,
                                                 pop_name,
                                                 attr_names[AttrMap::attr_index_uint16][i],
                                                 subset_uint16_values[i],
                                                 chunksize,
                                                 value_chunksize,
                                                 cachesize);
            }
          for (size_t i=0; i<subset_uint32_values.size(); i++)
            {
              write_cell_attribute_map<uint32_t>(all_comm,
                                                 output_file_name,
                                                 attr_namespace,
                                                 pop_name,
                                                 attr_names[AttrMap::attr_index_uint32][i],
                                                 subset_uint32_values[i],
                                                 chunksize,
                                                 value_chunksize,
                                                 cachesize);
            }
          
          for (size_t i=0; i<subset_int32_values.size(); i++)
            {
              write_cell_attribute_map<int32_t>(all_comm,
                                                output_file_name,
                                                attr_namespace,
                                                pop_name,
                                                attr_names[AttrMap::attr_index_int32][i],
                                                subset_int32_values[i],
                                                chunksize,
                                                value_chunksize,
                                                cachesize);
            }
        }
    }
  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  return 0;
}
