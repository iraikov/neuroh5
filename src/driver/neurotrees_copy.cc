// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_copy.cc
///
///  Driver program for copying tree structures.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include <mpi.h>
#include <hdf5.h>
#include <getopt.h>
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
#include <forward_list>

#include "throw_assert.hh"
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
#include "serialize_tree.hh"
#include "alltoallv_template.hh"


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
  cout << "Usage: " << string(argv[0]) << " [OPTIONS] <HDF FILE> <POPULATION> <SRC ID> <DEST ID>..." << endl <<
    "Options:" << endl <<
    "-h               Print this help" << endl <<
    "--fill           Copy the given source id to all cell ids in the population" << endl <<
    "--output FILE    Specify output file " << endl <<
    endl;
}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  int status;
  std::string pop_name;
  std::string input_filename, output_filename;
  size_t write_size=0;
  CELL_IDX_T source_gid;
  std::vector<CELL_IDX_T> target_gid_list;
  forward_list<neurotree_t> input_tree_list, output_tree_list;
  MPI_Comm all_comm;
  
  throw_assert(MPI_Init(&argc, &argv) >= 0,
               "error in MPI_Init");

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  throw_assert(MPI_Comm_size(all_comm, &size) == MPI_SUCCESS,
               "error in MPI_Comm_size");
  throw_assert(MPI_Comm_rank(all_comm, &rank) == MPI_SUCCESS,
               "error in MPI_Comm_size");

  int optflag_fill         = 0;
  int optflag_output       = 0;
  int optflag_write_size   = 0;
  bool opt_attributes      = false;
  bool opt_fill            = false;
  bool opt_output_filename = false;
  bool opt_write_size      = false;
  // parse arguments
  static struct option long_options[] = {
    {"fill",    no_argument, &optflag_fill,  1 },
    {"output",  required_argument, &optflag_output,  1 },
    {"write-size",  required_argument, &optflag_write_size,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "haow:", long_options, &option_index)) != -1)
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
          if (optflag_write_size == 1) {
            opt_write_size = true;
            optflag_write_size = 0;
            stringstream ss; 
            ss << string(string(optarg));
            ss >> write_size;
          }
          break;
        case 'a':
          opt_attributes = true;
          break;
        case 'o':
          opt_output_filename = true;
          output_filename = string(optarg);
          break;
        case 'w':
          opt_write_size = true;
          {
            stringstream ss; 
            ss << string(string(optarg));
            ss >> write_size;
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

  pop_range_map_t pop_ranges;
  size_t n_nodes;
  
  // Read population info
  throw_assert(cell::read_population_ranges(all_comm, input_filename, pop_ranges, n_nodes) >= 0,
               "error in read_population_ranges");

  pop_label_map_t pop_labels;
  throw_assert (cell::read_population_labels(all_comm, input_filename, pop_labels) >= 0,
                "error in read_population_labels");
                
  
  // Determine index of population to be read
  size_t pop_idx=0; bool pop_idx_set=false;
  for (auto& x : pop_labels)
    {
      if (get<1>(x) == pop_name)
        {
          pop_idx = get<0>(x);
          pop_idx_set = true;
        }
    }
  if (!pop_idx_set)
    {
      throw_err("Population not found");
    }

  CELL_IDX_T pop_start = pop_ranges[pop_idx].start;
  
  throw_assert(cell::read_trees (all_comm, input_filename,
                                 pop_name, pop_start,
                                 input_tree_list) >= 0,
               "error in read_trees");
  size_t local_tree_list_len = std::distance(input_tree_list.begin(), input_tree_list.end());
  size_t tree_list_len = 0;
  throw_assert(MPI_Reduce(&local_tree_list_len, &tree_list_len, 1, MPI_SIZE_T, MPI_SUM, 0, all_comm) == MPI_SUCCESS,
               "error in MPI_Reduce");

  if (rank == 0)
    {
      throw_assert(tree_list_len > 0,
                   "empty list of input trees");
    }
  
  for_each(input_tree_list.cbegin(),
           input_tree_list.cend(),
           [&] (const neurotree_t& tree)
           { cell::validate_tree(tree); } 
           );

  
  auto local_find_it = std::find_if(input_tree_list.begin(), input_tree_list.end(),
                                    [&](const neurotree_t& e) {return std::get<0>(e) == source_gid;});
  int find_result = local_find_it != input_tree_list.end();
  vector<int> find_results;
  find_results.resize(size, 0);
  throw_assert(MPI_Allgather(&find_result, 1, MPI_INT, &find_results[0], 1, MPI_INT, all_comm) == MPI_SUCCESS,
               "error in MPI_Allgather");
  auto find_it = std::find(find_results.begin(), find_results.end(), 1);
  throw_assert (find_it != find_results.end(),
                "unable to find source gid " << source_gid);
  rank_t found_rank_index = find_it - find_results.begin(); 

  map<rank_t, map<CELL_IDX_T, neurotree_t> > tree_rank_map;
  if (rank == found_rank_index)
    {
      const neurotree_t input_tree = *local_find_it;
      for (rank_t tree_rank = 0; tree_rank < size; tree_rank++)
        {
          tree_rank_map[tree_rank].insert(make_pair(source_gid, input_tree));
        }
    }


  std::forward_list<neurotree_t> source_tree_list;
  {
    // Create packed representation of the tree subset arranged per rank
    vector<char> sendbuf; 
    vector<int> sendcounts, sdispls;
    data::serialize_rank_tree_map (size, rank, tree_rank_map,
                                   sendcounts, sendbuf, sdispls);
    tree_rank_map.clear();
    
    // Send packed representation of the tree subset to the respective ranks
    vector<char> recvbuf; 
    vector<int> recvcounts, rdispls;
    throw_assert(mpi::alltoallv_vector(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                       recvcounts, rdispls, recvbuf) >= 0,
                 "neurotrees_copy: error while sending tree subset to assigned ranks"); 
    
    sendbuf.clear();
  
    // Unpack tree subset on the owning ranks
    data::deserialize_rank_tree_list (size, recvbuf, recvcounts, rdispls, source_tree_list);
    recvbuf.clear();
  }
  neurotree_t input_tree = *source_tree_list.begin();
      
  const deque<SECTION_IDX_T> & src_vector=get<1>(input_tree);
  const deque<SECTION_IDX_T> & dst_vector=get<2>(input_tree);
  const deque<SECTION_IDX_T> & sections=get<3>(input_tree);
  const deque<COORD_T> & xcoords=get<4>(input_tree);
  const deque<COORD_T> & ycoords=get<5>(input_tree);
  const deque<COORD_T> & zcoords=get<6>(input_tree);
  const deque<REALVAL_T> & radiuses=get<7>(input_tree);
  const deque<LAYER_IDX_T> & layers=get<8>(input_tree);
  const deque<PARENT_NODE_IDX_T> & parents=get<9>(input_tree);
  const deque<SWC_TYPE_T> & swc_types=get<10>(input_tree);
  

  if (opt_fill)
    {
      bool include_source_gid=false;
      if (input_filename.compare(output_filename) != 0)
        {
          include_source_gid=true;
        }
        
      target_gid_list.clear();
      for (size_t i=0; i<pop_ranges[pop_idx].count; i++)
        //for (size_t i=0; i<2000; i++)
        {
          if (i != source_gid-pop_start)
            {
              target_gid_list.push_back(i+pop_start);
            }
          else
            {
              if (include_source_gid)
                {
                  target_gid_list.push_back(i+pop_start);
                }
            }
        }
    }
  
  // determine which trees are written by which rank
  vector< pair<hsize_t,hsize_t> > ranges;
  mpi::rank_ranges(target_gid_list.size(), size, ranges);

  hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;
  
  if (access( output_filename.c_str(), F_OK ) != 0)
    {
      vector <string> groups;
      groups.push_back (hdf5::POPULATIONS);
      throw_assert(hdf5::create_file_toplevel (all_comm, output_filename, groups) == 0,
                   "error in create_file_toplevel");
    }

  if (rank == 0)
    {
      // TODO; create separate functions for opening HDF5 file for reading and writing
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert(fapl >= 0, "unable to create file access property list");
      hid_t output_file = H5Fopen(output_filename.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(output_file >= 0, "unable to open output file");
      
      if (!hdf5::exists_tree_h5types(output_file))
        {
          hid_t input_file = H5Fopen(input_filename.c_str(), H5F_ACC_RDONLY, fapl);
          throw_assert(input_file >= 0, "unable to open input file");
          status = hdf5::copy_tree_h5types(input_file, output_file);
          throw_assert(H5Fclose (input_file) == 0,
                       "unable to close input file");
        }

      throw_assert(H5Pclose (fapl) == 0, "unable to close file access property list");
      throw_assert(H5Fclose (output_file) == 0,
                   "unable to close output file");
    }
  MPI_Barrier(all_comm);


  if (write_size == 0)
    {
      write_size = end-start;
    }
  
  for (size_t i=start, ii=0; i<end; i++, ii++)
    {
      CELL_IDX_T gid = target_gid_list[i];
      CELL_IDX_T id = gid;
      printf("Task %d: Output local id %u (global id %u)\n", rank, id, gid);

      neurotree_t tree = make_tuple(id,
                                    src_vector, dst_vector, sections,
                                    xcoords, ycoords, zcoords,
                                    radiuses, layers, parents,
                                    swc_types);
      output_tree_list.push_front(tree);

      if ((ii > 0) && (ii % write_size == 0))
        {
          throw_assert(cell::append_trees(all_comm, output_filename, pop_name, pop_start, output_tree_list, size) == 0,
                       "error in append_trees");
          output_tree_list.clear();
        }

    }

  if (!output_tree_list.empty())
    {
      throw_assert(cell::append_trees(all_comm, output_filename, pop_name, pop_start, output_tree_list, size) == 0,
                   "error in append_trees");
    }

  MPI_Barrier(all_comm);
  MPI_Comm_free(&all_comm);

  MPI_Finalize();
  
  return status;
}
