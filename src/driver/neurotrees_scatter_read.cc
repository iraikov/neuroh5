// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_scatter_read.cc
///
///  Driver program for scatter_read_trees function.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <getopt.h>
#include <map>
#include <mpi.h>

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "cell_populations.hh"
#include "scatter_read_tree.hh"
#include "dataset_num_elements.hh"
#include "validate_tree.hh"
#include "attr_map.hh"
#include "tokenize.hh"
#include "throw_assert.hh"

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
  outfile.open(outfilename);

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

  outfile << "types: " << endl;
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

void summarize_tree(const string outfilename,
                    const CELL_IDX_T gid,
                    const neurotree_t& tree,
                    map <string, data::NamedAttrMap> &attr_maps)
{
  /*const std::vector<SECTION_IDX_T> & src_vector=get<1>(tree);*/
  /*const std::vector<SECTION_IDX_T> & dst_vector=get<2>(tree);*/
  const std::vector<SECTION_IDX_T> & sections=get<3>(tree);
  const std::vector<COORD_T> & xcoords=get<4>(tree);
  const std::vector<COORD_T> & ycoords=get<5>(tree);
  const std::vector<COORD_T> & zcoords=get<6>(tree);
  /*const std::vector<REALVAL_T> & radiuses=get<7>(tree);*/
  /*const std::vector<LAYER_IDX_T> & layers=get<8>(tree);*/
  /*const std::vector<PARENT_NODE_IDX_T> & parents=get<9>(tree);*/
  /*const std::vector<SWC_TYPE_T> & swc_types=get<10>(tree);*/

  ofstream fout;
  fout.open (outfilename, ios::app);
  fout << "gid: " << gid << endl;
  fout << "  number of x points: " << xcoords.size() << endl;
  fout << "  number of y points: " << ycoords.size() << endl;
  fout << "  number of z points: " << zcoords.size() << endl;
  fout << "  number of section points: " << sections.size() << endl;


  for (auto const& attr_map_entry : attr_maps)
    {
      const string& attr_name_space  = attr_map_entry.first;
      data::NamedAttrMap attr_map  = attr_map_entry.second;
      
      vector<vector<string>> attr_names;
      attr_map.attr_names(attr_names);

      const vector<vector<float>> &float_attrs     = attr_map.find<float>(gid);
      const vector<vector<uint8_t>> &uint8_attrs   = attr_map.find<uint8_t>(gid);
      const vector<vector<int8_t>> &int8_attrs     = attr_map.find<int8_t>(gid);
      const vector<vector<uint16_t>> &uint16_attrs = attr_map.find<uint16_t>(gid);
      const vector<vector<int16_t>>  &int16_attrs  = attr_map.find<int16_t>(gid);
      const vector<vector<uint32_t>> &uint32_attrs = attr_map.find<uint32_t>(gid);
      const vector<vector<int32_t>> &int32_attrs   = attr_map.find<int32_t>(gid);
      size_t attr_size = float_attrs.size() + 
        uint8_attrs.size() +
        int8_attrs.size() + 
        uint16_attrs.size() + 
        int16_attrs.size() + 
        uint32_attrs.size() + 
        int32_attrs.size();
      
      fout << "Attribute namespace " << attr_name_space << ": " << endl;
      fout << "  total number of attributes: " << attr_size << endl;
      
      for (size_t i=0; i<float_attrs.size(); i++)
        {
          fout << "  float attribute " << attr_names[data::AttrMap::attr_index_float][i] <<
        " is of size " << float_attrs[i].size() << endl;
        }
      for (size_t i=0; i<uint8_attrs.size(); i++)
        {
          fout << "  uint8 attribute " << attr_names[data::AttrMap::attr_index_uint8][i] <<
            " is of size " << uint8_attrs[i].size() << endl;
        }
      for (size_t i=0; i<int8_attrs.size(); i++)
        {
          fout << "  int8 attribute " << attr_names[data::AttrMap::attr_index_int8][i] <<
            " is of size " << int8_attrs[i].size() << endl;
        }
      for (size_t i=0; i<uint16_attrs.size(); i++)
        {
          fout << "  uint16 attribute " << attr_names[data::AttrMap::attr_index_uint16][i] <<
            " is of size " << uint16_attrs[i].size() << endl;
        }
      for (size_t i=0; i<uint32_attrs.size(); i++)
        {
          fout << "  uint32 attribute " << attr_names[data::AttrMap::attr_index_uint32][i] <<
            " is of size " << uint32_attrs[i].size() << endl;
        }
      for (size_t i=0; i<int32_attrs.size(); i++)
        {
          fout << "  int32 attribute " << attr_names[data::AttrMap::attr_index_int32][i] <<
            " is of size " << int32_attrs[i].size() << endl;
        }
    }
  fout.close();
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  herr_t status;
  MPI_Comm all_comm;
  std::string input_file_name, rank_file_name;
  vector<string> attr_name_spaces;
  size_t n_nodes;
  map<CELL_IDX_T, rank_t> node_rank_map;
  stringstream ss;

  throw_assert(MPI_Init(&argc, &argv) >= 0,
               "neurotrees_scatter_read: error in MPI initialization"); 

  int rank, size, io_size=1;
  throw_assert(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS,
               "neurotrees_scatter_read: error in MPI_Comm_size"); 
         
  throw_assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
               "neurotrees_scatter_read: error in MPI_Comm_rank"); 

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  debug_enabled = false;

  // parse arguments
  int optflag_verbose   = 0;
  int optflag_rankfile  = 0;
  int optflag_iosize    = 0;
  int optflag_namespace = 0;
  bool opt_rankfile = false,
    opt_iosize = false,
    opt_attrs = false,
    opt_namespace = false,
    opt_output = false;
  
  static struct option long_options[] = {
    {"verbose",   no_argument, &optflag_verbose,  1 },
    {"rankfile",  required_argument, &optflag_rankfile,  1 },
    {"iosize",    required_argument, &optflag_iosize,  1 },
    {"namespace", required_argument, &optflag_namespace,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "ar:i:n:oh",
                           long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_verbose == 1) {
            debug_enabled = true;
            optflag_verbose = 0;
          }
          if (optflag_rankfile == 1) {
            opt_rankfile = true;
            rank_file_name = std::string(strdup(optarg));
            optflag_rankfile = 0;
          }
          if (optflag_namespace == 1) {
            opt_namespace = true;
            string attr_name_space;
            string arg = string(optarg);
            string delimiter = ":";
            data::tokenize(arg, delimiter, attr_name_spaces);
            optflag_namespace = 0;
          }
          if (optflag_iosize == 1) {
            opt_iosize = true;
            ss << string(optarg);
            ss >> io_size;
            optflag_iosize = 0;
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
        case 'o':
          opt_output = true;
          break;
        case 'n':
          {
            opt_namespace = true;
            string attr_name_space;
            string arg = string(optarg);
            string delimiter = ":";
            data::tokenize(arg, delimiter, attr_name_spaces);
          }
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

  // Read population info to determine n_nodes
  throw_assert(cell::read_population_ranges(all_comm, input_file_name,
                                            pop_ranges, pop_vector,
                                            n_nodes) >= 0,
               "neurotrees_scatter_read: error in reading population ranges"); 

  vector<string> pop_names;
  status = cell::read_population_names(all_comm, input_file_name, pop_names);
  throw_assert (status >= 0,
                "neurotrees_scatter_read: error in reading population names"); 


  // Determine which nodes are assigned to which compute ranks
  if (!opt_rankfile)
    {
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < n_nodes; i++)
        {
          node_rank_map.insert(make_pair(i, i%size));
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

          throw_assert (iss >> n,
                        "neurotrees_scatter_read: invalid entry on node to rank assignment file"); 

          node_rank_map.insert(make_pair(i,n));
          i++;
        }

      infile.close();
    }

  map<CELL_IDX_T, neurotree_t>  tree_map;
  map<string, data::NamedAttrMap> attr_maps;
  
  for (size_t i = 0; i<pop_names.size(); i++)
    {
      if (rank == 0)
        printf("reading population %s...\n", pop_names[i].c_str());
      status = cell::scatter_read_trees (all_comm, input_file_name, io_size,
                                         attr_name_spaces, node_rank_map,
                                         pop_names[i], pop_vector[i].start,
                                         tree_map, attr_maps);

      
      throw_assert (status >= 0,
                    "neurotrees_scatter_read: error in reading trees"); 

      for_each(tree_map.cbegin(),
               tree_map.cend(),
               [&] (const pair<CELL_IDX_T, neurotree_t> &element)
               { const neurotree_t& tree = element.second;
                 cell::validate_tree(tree); } 
               );

    }
  
  size_t local_num_trees = tree_map.size();
  
  printf("Task %d has received a total of %lu trees\n", rank,  local_num_trees);
  
  if (opt_output)
    {
      if (tree_map.size() > 0)
        {
          for (auto const& element : tree_map)
            {
              const CELL_IDX_T gid = element.first;
              const neurotree_t &tree = element.second;
              stringstream outfilename;
              outfilename << string(input_file_name) << "." << gid << "." << rank
                          << ".trees";
              output_tree(outfilename.str(), tree);
            }
        }
    }
  else
    {
      stringstream outfilename_s;
      outfilename_s << string("neurotrees_scatter_read") << rank << ".dat";
      const string outfilename = outfilename_s.str();
      for (auto const& element : tree_map)
        {
          const CELL_IDX_T gid = element.first;
          const neurotree_t &tree = element.second;

          summarize_tree(outfilename, gid, tree, attr_maps);
        }
    }

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  return 0;
}
