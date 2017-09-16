// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurograph_scatter_read.cc
///
///  Driver program for scatter_graph function.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "tokenize.hh"
#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "read_graph.hh"
#include "scatter_read_graph.hh"
#include "projection_names.hh"

#include <mpi.h>

#include <getopt.h>

#include <cstdlib>
#include <cstring>
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
  printf("Usage: %s  [<OPTIONS>] <FILE> [<RANKFILE>]\n\n", argv[0]);
  printf("Options:\n");
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  std::string input_file_name, output_file_name, rank_file_name;
  // MPI Communicator for I/O ranks
  MPI_Comm all_comm;
  // A vector that maps nodes to compute ranks
  map<NODE_IDX_T, rank_t> node_rank_map;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
  vector<pair<string,string>> prj_names;
  vector < edge_map_t > prj_vector;
  vector < map <string, vector <vector<string> > > > edge_attr_name_vector;
  vector <string> edge_attr_name_spaces;
  stringstream ss;

  assert(MPI_Init(&argc, &argv) >= 0);

  EdgeMapType edge_map_type = EdgeMapDst;
  int rank, size, io_size; size_t n_nodes, local_num_nodes;
  size_t local_num_edges, total_num_edges;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  debug_enabled = false;

  // parse arguments
  int optflag_verbose = 0;
  int optflag_output = 0;
  int optflag_binary = 0;
  int optflag_rankfile = 0;
  int optflag_iosize = 0;
  int optflag_edgemap = 0;
  bool opt_binary = false,
    opt_rankfile = false,
    opt_iosize = false,
    opt_attrs = false,
    opt_output = false,
    opt_edgemap = false;

  static struct option long_options[] = {
    {"verbose",   no_argument, &optflag_verbose,  1 },
    {"output",    required_argument, &optflag_output,  1 },
    {"binary",    no_argument, &optflag_binary,  1 },
    {"rankfile",  required_argument, &optflag_rankfile,  1 },
    {"iosize",    required_argument, &optflag_iosize,  1 },
    {"edgemap",   required_argument, &optflag_edgemap,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "a:be:o:r:i:h",
                           long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_binary == 1) {
            opt_binary = true;
            optflag_binary = 0;
          }
          if (optflag_rankfile == 1) {
            opt_rankfile = true;
            rank_file_name = std::string(strdup(optarg));
            optflag_rankfile = 0;
          }
          if (optflag_iosize == 1) {
            opt_iosize = true;
            ss << string(optarg);
            ss >> io_size;
            optflag_iosize = 0;
          }
          if (optflag_verbose == 1) {
            debug_enabled = true;
            optflag_verbose = 0;
          }
          if (optflag_output == 1) {
            opt_output = true;
            output_file_name = std::string(strdup(optarg));
            optflag_output = 0;
          }
          if (optflag_edgemap == 1) {
            opt_edgemap = true;
            if (string(optarg) == "dst")
              {
                edge_map_type = EdgeMapDst;
              }
            if (string(optarg) == "src")
              {
                edge_map_type = EdgeMapSrc;
              }
            optflag_edgemap = 0;
          }
          break;
        case 'a':
          {
            opt_attrs = true;
            string arg = string(optarg);
            string namespace_delimiter = ",";
            tokenize(arg, namespace_delimiter, edge_attr_name_spaces);
          }
          break;
        case 'e':
          opt_edgemap = true;
          if ((string(optarg) == "dst") ||
              (string(optarg) == "destination"))
            {
              edge_map_type = EdgeMapDst;
            }
          if ((string(optarg) == "src") ||
              (string(optarg) == "source"))
            {
              edge_map_type = EdgeMapSrc;
            }
          break;
        case 'b':
          opt_binary = true;
          break;
        case 'h':
          print_usage_full(argv);
          exit(0);
          break;
        case 'i':
          opt_iosize = true;
          ss << string(optarg);
          ss >> io_size;
          break;
        case 'o':
          opt_output = true;
          output_file_name = std::string(strdup(optarg));
          break;
        case 'r':
          opt_rankfile = true;
          rank_file_name = std::string(strdup(optarg));
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if ((optind < argc) && opt_iosize)
    {
      input_file_name = std::string(argv[optind]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }


  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);

  // Read population info to determine n_nodes
  assert(cell::read_population_ranges(all_comm, input_file_name, pop_ranges,
                                      pop_vector, n_nodes) >= 0);

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
          rank_t n;

          assert (iss >> n);
          node_rank_map.insert(make_pair(i, n));
          i++;
        }

      infile.close();
    }

  DEBUG("scatter: reading projection names");

  assert(graph::read_projection_names(all_comm, input_file_name, prj_names) >= 0);
  MPI_Barrier(all_comm);
  DEBUG("scatter: finished reading projection names");

  
  DEBUG("scatter: calling scatter_graph");
  graph::scatter_read_graph (all_comm,
                             edge_map_type,
                             input_file_name,
                             io_size,
                             edge_attr_name_spaces,
                             prj_names,
                             node_rank_map,
                             prj_vector,
                             edge_attr_name_vector,
                             local_num_nodes,
                             n_nodes,
                             local_num_edges,
                             total_num_edges);


  if (opt_output)
    {
      if (opt_binary)
        {
          for (size_t i = 0; i < prj_vector.size(); i++)
            {
              DEBUG("scatter: outputting edges ", i);
              edge_map_t prj_edge_map = prj_vector[i];
              if (prj_edge_map.size() > 0)
                {
                  ofstream outfile;
                  stringstream outfilename;
                  outfilename << output_file_name << "." << i << "." <<
                    rank << ".edges.bin";
                  outfile.open(outfilename.str().c_str(), ios::binary);

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end();
                       it++)
                    {
                      NODE_IDX_T key_node   = it->first;
                      edge_tuple_t& et = it->second;

                      vector<NODE_IDX_T> adj_vector = get<0>(et);
                      const vector<data::AttrVal>& edge_attr_map = get<1>(et);

                      for (size_t j = 0; j < adj_vector.size(); j++)
                        {
                          switch (edge_map_type)
                            {
                            case EdgeMapDst:
                              {
                                NODE_IDX_T src = adj_vector[j];
                                NODE_IDX_T dst = key_node;
                                outfile << src << dst;
                              }
                              break;
                            case EdgeMapSrc:
                              {
                                NODE_IDX_T dst = adj_vector[j];
                                NODE_IDX_T src = key_node;
                                outfile << src << dst;
                              }
                              break;
                            }

                          for (const data::AttrVal& edge_attr_values : edge_attr_map)
                            {
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<float>(); k++)
                                {
                                  outfile << edge_attr_values.at<float>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint8_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<uint8_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint16_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<uint16_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint32_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<uint32_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int8_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<int8_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int16_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<int16_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int32_t>(); k++)
                                {
                                  outfile << edge_attr_values.at<int32_t>(k,j);
                                }
                            }

                        }
                    }
                  outfile.close();
                }
            }
        }
      else
        {
          for (size_t i = 0; i < prj_vector.size(); i++)
            {
              edge_map_t prj_edge_map = prj_vector[i];
              if (prj_edge_map.size() > 0)
                {
                  ofstream outfile;
                  stringstream outfilename;
                  outfilename << output_file_name << "." << i << "."
                              << rank << ".edges";
                  outfile.open(outfilename.str().c_str());

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end();
                       it++)
                    {
                      NODE_IDX_T key_node   = it->first;
                      edge_tuple_t& et = it->second;

                      const vector<NODE_IDX_T> adj_vector = get<0>(et);
                      const vector <data::AttrVal>& edge_attr_map = get<1>(et);

                      for (size_t j = 0; j < adj_vector.size(); j++)
                        {
                          
                          switch (edge_map_type)
                            {
                            case EdgeMapDst:
                              {
                                NODE_IDX_T src = adj_vector[j];
                                NODE_IDX_T dst = key_node;
                                outfile << src << " " << dst;
                              }
                              break;
                            case EdgeMapSrc:
                              {
                                NODE_IDX_T src = key_node;
                                NODE_IDX_T dst = adj_vector[i];
                                outfile << src << " " << dst;
                              }
                              break;
                            }

                          for (const data::AttrVal& edge_attr_values : edge_attr_map)
                            {
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<float>(); k++)
                                {
                                  outfile << " " << setprecision(9) <<
                                    edge_attr_values.at<float>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint8_t>(); k++)
                                {
                                  outfile << " " <<
                                    (unsigned int)edge_attr_values.at<uint8_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint16_t>(); k++)
                                {
                                  outfile << " " <<
                                    edge_attr_values.at<uint16_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<uint32_t>(); k++)
                                {
                                  outfile << " " <<
                                    edge_attr_values.at<uint32_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int8_t>(); k++)
                                {
                                  outfile << " " <<
                                    edge_attr_values.at<int8_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int16_t>(); k++)
                                {
                                  outfile << " " <<
                                    edge_attr_values.at<int16_t>(k,j);
                                }
                              for (size_t k = 0; k <
                                     edge_attr_values.size_attr_vec<int32_t>(); k++)
                                {
                                  outfile << " " <<
                                    edge_attr_values.at<int32_t>(k,j);
                                }
                            }

                          outfile << std::endl;
                        }
                    }
                  outfile.flush();
                  outfile.close();
                }
            }
        }
    }

  MPI_Barrier(all_comm);
  MPI_Comm_free(&all_comm);

  MPI_Finalize();
  return 0;
}
