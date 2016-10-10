
#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "population_reader.hh"
#include "graph_scatter.hh"

#include <hdf5.h>
#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include <mpi.h>

using namespace std;

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
            io_size = (size_t)std::stoi(string(optarg));;
          }
          if (optflag_nnodes == 1) {
            opt_nnodes = true;
            n_nodes = (size_t)std::stoi(string(optarg));
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
          io_size = (size_t)std::stoi(string(optarg));
          break;
        case 'n':
          opt_nnodes = true;
          n_nodes = (size_t)std::stoi(string(optarg));
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

  graph_scatter (all_comm,
                 input_file_name,
                 io_size,
                 opt_attrs,
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
                  outfile.open(outfilename.str(), ios::binary);

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
                    {
                      NODE_IDX_T dst   = it->first;
                      edge_tuple_t& et = it->second;
                      
                      vector<NODE_IDX_T> src_vect = get<0>(et);
                      vector<float>      longitudinal_distance_vect = get<1>(et);
                      vector<float>      transverse_distance_vect = get<2>(et);
                      vector<float>      distance_vect = get<3>(et);
                      vector<float>      synaptic_weight_vect = get<4>(et);
                      vector<uint16_t>   segment_index_vect = get<5>(et);
                      vector<uint16_t>   segment_point_index_vect = get<6>(et);
                      vector<uint8_t>    layer_vect = get<7>(et);
                      
                      for (size_t j = 0; j < src_vect.size(); j++)
                        {
                          NODE_IDX_T src = src_vect[j];
                          outfile << src << dst;
                          if (has_edge_attrs[0])
                            outfile << longitudinal_distance_vect[j];
                          if (has_edge_attrs[1])
                            outfile << transverse_distance_vect[j];
                          if (has_edge_attrs[2])
                            outfile << distance_vect[j];
                          if (has_edge_attrs[3])
                            outfile << synaptic_weight_vect[j];
                          if (has_edge_attrs[4])
                            outfile << segment_index_vect[j];
                          if (has_edge_attrs[5])
                            outfile << segment_point_index_vect[j];
                          if (has_edge_attrs[6])
                            outfile << layer_vect[j];
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
                  outfile.open(outfilename.str());

                  for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
                    {
                      NODE_IDX_T dst   = it->first;
                      edge_tuple_t& et = it->second;

                      const vector<NODE_IDX_T>  src_vect                   = get<0>(et);
                      const vector<float>&      longitudinal_distance_vect = get<1>(et);
                      const vector<float>&      transverse_distance_vect   = get<2>(et);
                      const vector<float>&      distance_vect              = get<3>(et);
                      const vector<float>&      synaptic_weight_vect       = get<4>(et);
                      const vector<uint16_t>&   segment_index_vect         = get<5>(et);
                      const vector<uint16_t>&   segment_point_index_vect   = get<6>(et);
                      const vector<uint8_t>&    layer_vect                 = get<7>(et);

                      for (size_t j = 0; j < src_vect.size(); j++)
                        {
                          NODE_IDX_T src = src_vect[j];
                          outfile << "    " << src << " " << dst;
                          if (has_edge_attrs[0])
                            outfile << " " << longitudinal_distance_vect[j];
                          if (has_edge_attrs[1])
                            outfile << " " << transverse_distance_vect[j];
                          if (has_edge_attrs[2])
                            outfile << " " << distance_vect[j];
                          if (has_edge_attrs[3])
                            outfile << " " << synaptic_weight_vect[j];
                          if (has_edge_attrs[4])
                            outfile << " " << segment_index_vect[j];
                          if (has_edge_attrs[5])
                            outfile << " " << segment_point_index_vect[j];
                          if (has_edge_attrs[6])
                            outfile << " " << layer_vect[j];
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
