// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_select.cc
///
///  Program for selecting tree subsets.
///
///  Copyright (C) 2016-2021 Project Neurotrees.
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
#include <forward_list>
#include <getopt.h>
#include <mpi.h>

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "scatter_read_tree.hh"
#include "append_tree.hh"
#include "cell_attributes.hh"
#include "cell_populations.hh"
#include "serialize_tree.hh"
#include "alltoallv_template.hh"
#include "sort_permutation.hh"
#include "attr_map.hh"
#include "tokenize.hh"
#include "dataset_num_elements.hh"
#include "validate_tree.hh"
#include "create_file_toplevel.hh"
#include "exists_tree_h5types.hh"
#include "copy_tree_h5types.hh"
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
  printf("Usage: %s [treefile] [options] INPUT_FILE SELECTION_NAMESPACE OUTPUT_FILE\n\n", argv[0]);
  printf("Options:\n");
  printf("\t--verbose:\n");
  printf("\t\tPrint verbose diagnostic information\n");
}



// Assign each cell ID to a rank 
void compute_node_rank_map
(
 size_t num_ranks,
 vector <CELL_IDX_T> index_vector, 
 node_rank_map_t &node_rank_map
 )
{
  hsize_t remainder=0, offset=0, buckets=0;
  size_t num_nodes = index_vector.size();
  
  for (size_t i=0; i<num_ranks; i++)
    {
      remainder  = num_nodes - offset;
      buckets    = num_ranks - i;
      for (size_t j = 0; j < remainder / buckets; j++)
        {
          node_rank_map[index_vector[offset+j]].insert(i);
        }
      offset    += remainder / buckets;
    }
}

/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  herr_t status;
  MPI_Comm all_comm;
  string pop_name, input_file_name, output_file_name, selection_namespace, selection_attr, rank_file_name;
  vector<string> attr_name_spaces;
  size_t n_nodes;
  node_rank_map_t node_rank_map;
  stringstream ss;

  throw_assert(MPI_Init(&argc, &argv) >= 0,
               "neurotrees_select: error in MPI initialization"); 

  size_t chunksize=1000, value_chunksize=1000, cachesize=1*1024*1024;
  int rank, size, io_size=1;
  throw_assert(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS,
               "neurotrees_select: error in MPI_Comm_size"); 
  throw_assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
               "neurotrees_select: error in MPI_Comm_rank"); 

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  debug_enabled = false;

  // parse arguments
  int optflag_verbose   = 0;
  int optflag_rankfile  = 0;
  int optflag_reindex   = 0;
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
    opt_cachesize   = false,
    opt_reindex     = false;
  
  static struct option long_options[] = {
    {"verbose",   no_argument, &optflag_verbose,  1 },
    {"rankfile",  required_argument, &optflag_rankfile,  1 },
    {"reindex",   no_argument, &optflag_reindex,  1 },
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
            optflag_rankfile = 0;
          }
          if (optflag_reindex == 1) {
            opt_reindex = true;
            optflag_reindex = 0;
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
          if (optflag_chunksize == 1) {
            opt_chunksize = true;
            ss << string(optarg);
            ss >> chunksize;
            optflag_chunksize = 0;
          }
          if (optflag_value_chunksize == 1) {
            opt_value_chunksize = true;
            ss << string(optarg);
            ss >> value_chunksize;
            optflag_value_chunksize = 0;
          }
          if (optflag_cachesize == 1) {
            opt_cachesize = true;
            ss << string(optarg);
            ss >> cachesize;
            optflag_cachesize = 0;
          }
          if (optflag_verbose == 1) {
            debug_enabled = true;
            optflag_verbose = 0;
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
          {
            opt_namespace = true;
            string attr_name_space;
            string arg = string(optarg);
            string delimiter = ":";
            data::tokenize(arg, delimiter, attr_name_spaces);
          }
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
      string selection_delimiter = "/";
      vector <string> selection_spec;
      data::tokenize(string(argv[optind+1]), selection_delimiter, selection_spec);
      selection_namespace = selection_spec[0];
      if (selection_spec.size() > 1)
        {
          selection_attr = selection_spec[1];
        }
      else
        {
          selection_attr = "New Cell Index";
        }
      output_file_name = string(argv[optind+2]);
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  pop_range_map_t pop_ranges;

  // Read population info to determine n_nodes
  throw_assert(cell::read_population_ranges(all_comm, input_file_name, pop_ranges, n_nodes) >= 0,
               "neurotrees_select: error in reading population ranges"); 

  pop_label_map_t pop_labels;
  status = cell::read_population_labels(all_comm, input_file_name, pop_labels);
  throw_assert (status >= 0,
                "neurotrees_select: error in reading population labels"); 

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

  size_t pop_start = pop_ranges[pop_idx].start;

  // Read in selection indices
  set<CELL_IDX_T> tree_selection;
  map<CELL_IDX_T, CELL_IDX_T> selection_map;
  set<string> attr_mask;

  {
    data::NamedAttrMap selection_attr_map;
    cell::bcast_cell_attributes (all_comm, 0, input_file_name, selection_namespace, attr_mask,
                                 pop_name, pop_start, selection_attr_map);

    auto name_map = selection_attr_map.attr_name_map[typeid(uint32_t)];
    auto const& name_it = name_map.find(selection_attr);
    throw_assert(name_it != name_map.end(),
                 "neurotrees_select: selection attribute not found"); 

    size_t newgid_attr_index = name_it->second;
    auto attr_map = selection_attr_map.attr_map<uint32_t>(newgid_attr_index);
    for (auto const& attr_it : attr_map)
       {
         selection_map.insert(make_pair(attr_it.first, attr_it.second[0]));
       }
  }

  // Determine which nodes are assigned to which compute ranks
  if (!opt_rankfile)
    {
      // round-robin node to rank assignment from file
      for (size_t i = 0; i < n_nodes; i++)
        {
          node_rank_map[i].insert(i%size);
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
                        "neurotrees_select: invalid entry in node to rank assignment file"); 

          node_rank_map[i].insert(n);
          i++;
        }

      infile.close();
    }

  // Compute an assignment of subset trees to IO ranks
  node_rank_map_t subset_node_rank_map;
  {
    vector<CELL_IDX_T> selection_index;
    for (auto const& element : selection_map)
      {
        selection_index.push_back(element.second);
      }

    auto compare_idx = [](const CELL_IDX_T& a, const CELL_IDX_T& b) { return (a < b); };
    vector<size_t> p = data::sort_permutation(selection_index, compare_idx);
    data::apply_permutation_in_place(selection_index, p);

    compute_node_rank_map(io_size,
                          selection_index,
                          subset_node_rank_map);
  }

  
  map<CELL_IDX_T, neurotree_t>  tree_map;
  map<string, data::NamedAttrMap> attr_maps;
  map<string, vector<vector<string> > > attr_names_map;

  std::forward_list<neurotree_t> tree_subset;
  map<rank_t, map<CELL_IDX_T, neurotree_t> > tree_subset_rank_map;


  map <string, vector<map< CELL_IDX_T, deque<float> > > >
    subset_float_value_map;
  map <string, vector<map< CELL_IDX_T, deque<uint8_t> > > >
    subset_uint8_value_map;
  map <string, vector<map< CELL_IDX_T, deque<int8_t> > > >
    subset_int8_value_map;
  map <string, vector<map< CELL_IDX_T, deque<uint16_t> > > >
    subset_uint16_value_map;
  map <string, vector<map< CELL_IDX_T, deque<int16_t> > > >
    subset_int16_value_map;
  map <string, vector<map< CELL_IDX_T, deque<uint32_t> > > >
    subset_uint32_value_map;
  map <string, vector<map< CELL_IDX_T, deque<int32_t> > > >
    subset_int32_value_map;

  
  status = cell::scatter_read_trees (all_comm, input_file_name, io_size,
                                     attr_name_spaces, node_rank_map,
                                     pop_name, pop_ranges[pop_idx].start,
                                     tree_map, attr_maps);
  
  
  throw_assert (status >= 0,
                "neurotrees_select: error in reading trees"); 

  
  for_each(tree_map.cbegin(),
           tree_map.cend(),
           [&] (const pair<CELL_IDX_T, neurotree_t> &element)
           { const neurotree_t& tree = element.second;
             cell::validate_tree(tree); } 
           );
  
  size_t local_num_trees = tree_map.size();
  
  printf("Task %d has received a total of %lu trees\n", rank,  local_num_trees);
  
  for (auto & element : tree_map)
    {
      const CELL_IDX_T idx = element.first;
      if (tree_selection.find(idx) != tree_selection.end())
        {
          neurotree_t &tree = element.second;
          auto it = selection_map.find(idx);
          throw_assert(it != selection_map.end(),
                       "neurotrees_select: selection tree index not found"); 

          CELL_IDX_T idx1 = it->second;
          get<0>(tree) = idx1;
          tree_subset.push_front(tree);

          for (auto const& attr_map_entry : attr_maps)
            {
              const string& attr_name_space  = attr_map_entry.first;
              data::NamedAttrMap attr_map  = attr_map_entry.second;

              vector <size_t> num_attrs;
              num_attrs.resize(data::AttrMap::num_attr_types);
          
              vector<vector<string>> attr_names;
              attr_names.resize(data::AttrMap::num_attr_types);
          
              attr_map.attr_names(attr_names);
              attr_map.num_attrs(num_attrs);

              attr_names_map.insert(make_pair(attr_name_space, attr_names));
              
              const vector<deque<float>>    float_values  = attr_map.find<float>(idx);
              const vector<deque<uint8_t>>  uint8_values  = attr_map.find<uint8_t>(idx);
              const vector<deque<int8_t>>   int8_values   = attr_map.find<int8_t>(idx);
              const vector<deque<uint16_t>> uint16_values = attr_map.find<uint16_t>(idx);
              const vector<deque<int16_t>>  int16_values = attr_map.find<int16_t>(idx);
              const vector<deque<uint32_t>> uint32_values = attr_map.find<uint32_t>(idx);
              const vector<deque<int32_t>>  int32_values  = attr_map.find<int32_t>(idx);

              vector<map< CELL_IDX_T, deque<float> > > 
                subset_float_values(num_attrs[data::AttrMap::attr_index_float]);
              vector<map< CELL_IDX_T, deque<uint8_t> > > 
                subset_uint8_values(num_attrs[data::AttrMap::attr_index_uint8]);
              vector<map< CELL_IDX_T, deque<int8_t> > > 
                subset_int8_values(num_attrs[data::AttrMap::attr_index_int8]);
              vector<map< CELL_IDX_T, deque<uint16_t> > > 
                subset_uint16_values(num_attrs[data::AttrMap::attr_index_uint16]);
              vector<map< CELL_IDX_T, deque<int16_t> > > 
                subset_int16_values(num_attrs[data::AttrMap::attr_index_int16]);
              vector<map< CELL_IDX_T, deque<uint32_t> > > 
                subset_uint32_values(num_attrs[data::AttrMap::attr_index_uint32]);
              vector<map< CELL_IDX_T, deque<int32_t> > > 
                subset_int32_values(num_attrs[data::AttrMap::attr_index_int32]);
              
              for (size_t i=0; i<float_values.size(); i++)
                {
                  subset_float_values[i].insert(make_pair(idx1, deque<float>(float_values[i].begin(), float_values[i].end())));
                }
              for (size_t i=0; i<uint8_values.size(); i++)
                {
                  subset_uint8_values[i].insert(make_pair(idx1, deque<uint8_t>(uint8_values[i].begin(), uint8_values[i].end())));
                }
              for (size_t i=0; i<int8_values.size(); i++)
                {
                  subset_int8_values[i].insert(make_pair(idx1, deque<int8_t>(int8_values[i].begin(), int8_values[i].end())));
                }
              for (size_t i=0; i<uint16_values.size(); i++)
                {
                  subset_uint16_values[i].insert(make_pair(idx1, deque<uint16_t>(uint16_values[i].begin(), uint16_values[i].end())));
                }
              for (size_t i=0; i<int16_values.size(); i++)
                {
                  subset_int16_values[i].insert(make_pair(idx1, deque<int16_t>(int16_values[i].begin(), int16_values[i].end())));
                }
              for (size_t i=0; i<uint32_values.size(); i++)
                {
                  subset_uint32_values[i].insert(make_pair(idx1, deque<uint32_t>(uint32_values[i].begin(), uint32_values[i].end())));
                }
              for (size_t i=0; i<int32_values.size(); i++)
                {
                  subset_int32_values[i].insert(make_pair(idx1, deque<int32_t>(int32_values[i].begin(), int32_values[i].end())));
                }

              subset_float_value_map[attr_name_space]  = subset_float_values;
              subset_uint8_value_map[attr_name_space]  = subset_uint8_values;
              subset_int8_value_map[attr_name_space]   = subset_int8_values;
              subset_uint16_value_map[attr_name_space] = subset_uint16_values;
              subset_int16_value_map[attr_name_space]  = subset_int16_values;
              subset_uint32_value_map[attr_name_space] = subset_uint32_values;
              subset_int32_value_map[attr_name_space]  = subset_int32_values;

            }
        }
    }

  tree_map.clear();

  uint32_t global_subset_size=0, local_subset_size=0;

  for (auto const& tree : tree_subset)
    {
      CELL_IDX_T idx = get<0>(tree);

      auto it = subset_node_rank_map.find(idx);
      throw_assert(it != subset_node_rank_map.end(),
                   "neurotrees_select: tree index not found in node rank assignment"); 

      set<rank_t> tree_ranks = it->second;
      for (rank_t tree_rank : tree_ranks)
        {
          tree_subset_rank_map[tree_rank].insert(make_pair(idx, tree));
        }

      local_subset_size++;
    }
  subset_node_rank_map.clear();
  tree_subset.clear();
  

  // Determine the total number of selected trees
  throw_assert(MPI_Reduce(&local_subset_size, &global_subset_size, 1, MPI_UINT32_T,
                    MPI_SUM, 0, all_comm) == MPI_SUCCESS,
         "neurotrees_select: error in MPI_Reduce"); 

  throw_assert(MPI_Bcast(&global_subset_size, 1, MPI_UINT32_T, 0, all_comm) == MPI_SUCCESS,
               "neurotrees_select: error in MPI_Bcast"); 


  // Created packed representation of the tree subset arranged per rank
  vector<char> sendbuf; 
  vector<int> sendcounts, sdispls;
  data::serialize_rank_tree_map (size, rank, tree_subset_rank_map,
                                 sendcounts, sendbuf, sdispls);
  tree_subset_rank_map.clear();
  
  // Send packed representation of the tree subset to the respective ranks
  vector<char> recvbuf; 
  vector<int> recvcounts, rdispls;
  throw_assert(mpi::alltoallv_vector(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                     recvcounts, rdispls, recvbuf) >= 0,
               "neurotrees_select: error while sending tree subset to assigned ranks"); 

  sendbuf.clear();

  // Unpack tree subset on the owning ranks
  data::deserialize_rank_tree_list (size, recvbuf, recvcounts, rdispls, tree_subset);
  recvbuf.clear();

  auto compare_trees = [](const neurotree_t& a, const neurotree_t& b) { return (get<0>(a) < get<0>(b)); };
  tree_subset.sort(compare_trees);
  
  if (global_subset_size > 0)
    {
      //status = access( output_file_name.c_str(), F_OK );
      hid_t input_file;
      vector <string> groups;
      groups.push_back (hdf5::POPULATIONS);
      status = hdf5::create_file_toplevel (all_comm, output_file_name, groups);
      throw_assert(status == 0,
                   "neurotrees_select: error in creating output HDF5 file"); 

      MPI_Barrier(all_comm);
      
      // TODO; create separate functions for opening HDF5 file for reading and writing
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
#ifdef HDF5_IS_PARALLEL
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, all_comm, MPI_INFO_NULL) >= 0);
#endif
      hid_t output_file = H5Fopen(output_file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(output_file >= 0,
                   "neurotrees_select: error in opening output HDF5 file"); 
      
      if (!hdf5::exists_tree_h5types(output_file))
        {
          input_file = H5Fopen(input_file_name.c_str(), H5F_ACC_RDONLY, fapl);
          throw_assert(input_file >= 0,
                       "neurotrees_select: error in opening input HDF5 file with H5Types definition"); 

          status = hdf5::copy_tree_h5types(input_file, output_file);
          status = H5Fclose (input_file);
          throw_assert(status == 0,
                       "neurotrees_select: error in closing input HDF5 file with H5Types definition"); 

        }

      status = H5Pclose (fapl);
      throw_assert_nomsg(status == 0);
      status = H5Fclose (output_file);
      throw_assert(status == 0,
                   "neurotrees_select: error in closing output file");

      pop_range_map_t output_pop_ranges;

      // Read population info to determine n_nodes
      throw_assert(cell::read_population_ranges(all_comm, output_file_name,
                                                output_pop_ranges, 
                                                n_nodes) >= 0,
                   "neurotrees_select: error in reading population ranges"); 

      pop_label_map_t output_pop_labels;
      status = cell::read_population_labels(all_comm, output_file_name, output_pop_labels);
      throw_assert (status >= 0,
                    "neurotrees_select: error in reading population labels"); 
      
      // Determine index of population to be read
      size_t output_pop_idx=0; bool output_pop_idx_set=false;
      for (auto & x : output_pop_labels)
        {
          if (get<1>(x) == pop_name)
            {
              output_pop_idx = get<0>(x);
              output_pop_idx_set = true;
            }
        }
      if (!output_pop_idx_set)
        {
          throw_err("Population not found");
        }
      
      size_t output_pop_start = output_pop_ranges[output_pop_idx].start;

      status = cell::append_trees(all_comm, output_file_name, pop_name, output_pop_start, tree_subset, size);
      
      throw_assert(status == 0,
                   "neurotrees_select: error in appending trees to output file"); 

      for (string& attr_name_space : attr_name_spaces)
        {
          const vector<map< CELL_IDX_T, deque<float> > > &
            subset_float_values = subset_float_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<uint8_t> > > &
            subset_uint8_values = subset_uint8_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<int8_t> > > 
            subset_int8_values = subset_int8_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<uint16_t> > > 
            subset_uint16_values = subset_uint16_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<int16_t> > > 
            subset_int16_values = subset_int16_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<uint32_t> > > 
            subset_uint32_values = subset_uint32_value_map[attr_name_space];
          const vector<map< CELL_IDX_T, deque<int32_t> > > 
            subset_int32_values = subset_int32_value_map[attr_name_space];

          const vector<vector<string>>& attr_names = attr_names_map[attr_name_space];
          const data::optional_hid dflt_data_type;
          
          for (size_t i=0; i<subset_float_values.size(); i++)
            {
              cell::write_cell_attribute_map<float>(all_comm,
                                                    output_file_name,
                                                    attr_name_space,
                                                    pop_name, pop_start,
                                                    attr_names[data::AttrMap::attr_index_float][i],
                                                    subset_float_values[i],
                                                    io_size,
                                                    dflt_data_type,
                                                    IndexOwner,
                                                    CellPtr(PtrOwner),
                                                    chunksize,
                                                    value_chunksize,
                                                    cachesize);
            }
          for (size_t i=0; i<subset_uint8_values.size(); i++)
            {
              cell::write_cell_attribute_map<uint8_t>(all_comm,
                                                      output_file_name,
                                                      attr_name_space,
                                                      pop_name, pop_start,
                                                      attr_names[data::AttrMap::attr_index_uint8][i],
                                                      subset_uint8_values[i],
                                                      io_size,
                                                      dflt_data_type,
                                                      IndexOwner,
                                                      CellPtr(PtrOwner),
                                                      chunksize,
                                                      value_chunksize,
                                                      cachesize);
            }
          for (size_t i=0; i<subset_int8_values.size(); i++)
            {
              cell::write_cell_attribute_map<int8_t>(all_comm,
                                                     output_file_name,
                                                     attr_name_space,
                                                     pop_name, pop_start,
                                                     attr_names[data::AttrMap::attr_index_int8][i],
                                                     subset_int8_values[i],
                                                     io_size,
                                                     dflt_data_type,
                                                     IndexOwner,
                                                     CellPtr(PtrOwner),
                                                     chunksize,
                                                     value_chunksize,
                                                     cachesize);
            }
          for (size_t i=0; i<subset_uint16_values.size(); i++)
            {
              cell::write_cell_attribute_map<uint16_t>(all_comm,
                                                       output_file_name,
                                                       attr_name_space,
                                                       pop_name, pop_start,
                                                       attr_names[data::AttrMap::attr_index_uint16][i],
                                                       subset_uint16_values[i],
                                                       io_size,
                                                       dflt_data_type,
                                                       IndexOwner,
                                                       CellPtr(PtrOwner),
                                                       chunksize,
                                                       value_chunksize,
                                                       cachesize);
            }
          for (size_t i=0; i<subset_uint32_values.size(); i++)
            {
              cell::write_cell_attribute_map<uint32_t>(all_comm,
                                                       output_file_name,
                                                       attr_name_space,
                                                       pop_name, pop_start,
                                                       attr_names[data::AttrMap::attr_index_uint32][i],
                                                       subset_uint32_values[i],
                                                       io_size,
                                                       dflt_data_type,
                                                       IndexOwner,
                                                       CellPtr(PtrOwner),
                                                       chunksize,
                                                       value_chunksize,
                                                       cachesize);
            }
          
          for (size_t i=0; i<subset_int32_values.size(); i++)
            {
              cell::write_cell_attribute_map<int32_t>(all_comm,
                                                      output_file_name,
                                                      attr_name_space,
                                                      pop_name, pop_start,
                                                      attr_names[data::AttrMap::attr_index_int32][i],
                                                      subset_int32_values[i],
                                                      io_size,
                                                      dflt_data_type,
                                                      IndexOwner,
                                                      CellPtr(PtrOwner),
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
