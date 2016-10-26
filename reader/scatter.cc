
#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "population_reader.hh"
#include "graph_reader.hh"

#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <hdf5.h>
#include <mpi.h>

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
  printf("Usage: %s  <FILE> <N> <IOSIZE> [<RANKFILE>]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-s:\n");
  printf("\t\tPrint only edge summary\n");
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  char *input_file_name, *output_file_name, *rank_file_name;
  // MPI Communicator for I/O ranks
  MPI_Comm all_comm;
  // A vector that maps nodes to compute ranks
  vector<rank_t> node_rank_vector;
  // The set of compute ranks for which the current I/O rank is responsible
  rank_edge_map_t rank_edge_map;
  set< pair<pop_t, pop_t> > pop_pairs;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  vector<string> prj_names;
  vector < edge_map_t > prj_vector;
  vector < vector<uint8_t> > has_edge_attrs_vector;
  stringstream ss;

  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size, io_size; size_t n_nodes;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments
  int optflag_output = 0;
  int optflag_binary = 0;
  int optflag_rankfile = 0;
  int optflag_iosize = 0;
  int optflag_nnodes = 0;
  bool opt_binary = false,
    opt_rankfile = false,
    opt_iosize = false,
    opt_nnodes = false,
    opt_attrs = false,
    opt_output = false;

  static struct option long_options[] = {
    {"output",    required_argument, &optflag_output,  1 },
    {"binary",    no_argument, &optflag_binary,  1 },
    {"rankfile",  required_argument, &optflag_rankfile,  1 },
    {"iosize",    required_argument, &optflag_iosize,  1 },
    {"nnodes",    required_argument, &optflag_nnodes,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "abr:i:n:h",
			   long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_binary == 1) {
            opt_binary = true;
          }
          if (optflag_rankfile == 1) {
            opt_rankfile = true;
            rank_file_name = strdup(optarg);
          }
          if (optflag_iosize == 1) {
            opt_iosize = true;
	    ss << string(optarg);
	    ss >> io_size;
          }
          if (optflag_nnodes == 1) {
            opt_nnodes = true;
	    ss << string(optarg);
	    ss >> n_nodes;
          }
          break;
        case 'a':
          opt_attrs = true;
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
        case 'n':
          opt_nnodes = true;
	  ss << string(optarg);
	  ss >> n_nodes;
          break;
        case 'o':
          opt_output = true;
          output_file_name = strdup(optarg);
          break;
        case 'r':
          opt_rankfile = true;
          rank_file_name = strdup(optarg);
          break;
        default:
          throw_err("Input argument format error");
        }
    }

  if ((optind < argc) && (opt_nnodes || opt_rankfile) && opt_iosize)
    {
      input_file_name = argv[optind];
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  // Determine which nodes are assigned to which compute ranks
  node_rank_vector.resize(n_nodes);
  if (!opt_rankfile)
    {
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < n_nodes; i++)
        {
          node_rank_vector[i] = i%size;
        }
    }
  else
    {
      ifstream infile(rank_file_name);
      string line;
      size_t i = 0;
      // reads node to rank assignment from file
      while (getline(infile, line))
        {
          istringstream iss(line);
          rank_t n;
          
          assert (iss >> n);
          node_rank_vector[i] = n;
          i++;
        }
      
      infile.close();
    }


  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);

  vector<string> prj_names;
  assert(read_projection_names(all_comm, input_file_name, prj_names) >= 0);

  
  scatter_graph (all_comm,
                 input_file_name,
                 io_size,
                 opt_attrs,
                 prj_names,
                 node_rank_vector,
                 prj_vector,
                 has_edge_attrs_vector);


  if (opt_output)
    {
      if (opt_binary)
        {
          for (size_t i = 0; i < prj_vector.size(); i++)
            {
              DEBUG("scatter: outputting edges ", i);
              edge_map_t prj_edge_map = prj_vector[i];
              vector <uint8_t> has_edge_attrs = has_edge_attrs_vector[i];
              if (prj_edge_map.size() > 0)
                {
                  ofstream outfile;
                  stringstream outfilename;
                  outfilename << string(output_file_name) << "." << i << "." << rank << ".edges.bin";
                  outfile.open(outfilename.str().c_str(), ios::binary);

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
                    {
                      NODE_IDX_T dst   = it->first;
                      edge_tuple_t& et = it->second;
                      
                      vector<NODE_IDX_T> src_vect = get<0>(et);
                      const EdgeAttr&   edge_attr_values = get<1>(et);
                      
                      for (size_t j = 0; j < src_vect.size(); j++)
                        {
                          NODE_IDX_T src = src_vect[j];
                          outfile << src << dst;
                          for (size_t k = 0; k < edge_attr_values.size<float>(); k++)
                            {
                              outfile << edge_attr_values.at<float>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint8_t>(); k++)
                            {
                              outfile << edge_attr_values.at<uint8_t>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint16_t>(); k++)
                            {
                              outfile << edge_attr_values.at<uint16_t>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint32_t>(); k++)
                            {
                              outfile << edge_attr_values.at<uint32_t>(k,j); 
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
              vector <uint8_t> has_edge_attrs = has_edge_attrs_vector[i];
              if (prj_edge_map.size() > 0)
                {
                  ofstream outfile;
                  stringstream outfilename;
                  outfilename << string(output_file_name) << "." << i << "." << rank << ".edges";
                  outfile.open(outfilename.str().c_str());

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
                    {
                      NODE_IDX_T dst   = it->first;
                      edge_tuple_t& et = it->second;

                      const vector<NODE_IDX_T> src_vect = get<0>(et);
                      const EdgeAttr&   edge_attr_values = get<1>(et);

                      
                      for (size_t j = 0; j < src_vect.size(); j++)
                        {
                          NODE_IDX_T src = src_vect[j];
                          outfile << "    " << src << " " << dst;
                          for (size_t k = 0; k < edge_attr_values.size<float>(); k++)
                            {
                              outfile << " " << edge_attr_values.at<float>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint8_t>(); k++)
                            {
                              outfile << " " << edge_attr_values.at<uint8_t>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint16_t>(); k++)
                            {
                              outfile << " " << edge_attr_values.at<uint16_t>(k,j); 
                            }
                          for (size_t k = 0; k < edge_attr_values.size<uint32_t>(); k++)
                            {
                              outfile << " " << edge_attr_values.at<uint32_t>(k,j); 
                            }
                          
                          outfile << std::endl;
                        }
                    }
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
