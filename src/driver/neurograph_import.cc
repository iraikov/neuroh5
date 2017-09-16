// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurograph_import.cc
///
///  Driver program for various import procedures.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "projection_names.hh"
#include "read_syn_projection.hh"
#include "read_txt_projection.hh"
#include "rank_range.hh"
#include "write_graph.hh"
#include "attr_map.hh"
#include "attr_val.hh"
#include "tokenize.hh"

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
  printf("Usage: %s  <SRC-POP> <DST-POP> <OUTPUT-FILE> \n\n", argv[0]);
  printf("Options:\n");
  printf("\t-i <FILE>:\n");
  printf("\t\tImport from given file\n");
  printf("\t-f <FORMAT>:\n");
  printf("\t\tInput format\n");

}
  

int append_syn_adj_map
(
 const vector<NODE_IDX_T>&   src_range,
 const int src_offset, const int dst_offset,
 const vector<NODE_IDX_T>&  dst_idx,
 const vector<DST_PTR_T>&   src_idx_ptr,
 const vector<NODE_IDX_T>&  src_idx,
 const vector<DST_PTR_T>&   syn_idx_ptr,
 const vector<NODE_IDX_T>&  syn_idx,
 size_t&                    num_edges,
 edge_map_t&                edge_map
 )
{
  int ierr = 0; 
  num_edges = 0;

  if (dst_idx.size() > 0)
    {
      for (size_t d = 0; d < dst_idx.size()-1; d++)
        {
          NODE_IDX_T dst = dst_idx[d] + dst_offset;

          size_t low_src_ptr = src_idx_ptr[d],
            high_src_ptr = src_idx_ptr[d+1];
          size_t low_syn_ptr = syn_idx_ptr[d],
            high_syn_ptr = syn_idx_ptr[d+1];

          vector<NODE_IDX_T> adj_vector;
          vector <data::AttrVal> edge_attr_values(1);
          vector<NODE_IDX_T> syn_id_vector;

          for (size_t i = low_src_ptr, ii = low_syn_ptr; i < high_src_ptr; ++i, ++ii)
            {
              assert(ii < high_syn_ptr);
              NODE_IDX_T src = src_idx[i];
              if (src <= src_range[1] && src >= src_range[0])
                {
                  NODE_IDX_T src1 = src + src_offset;
                  NODE_IDX_T syn_id = syn_idx[ii];
                  adj_vector.push_back(src1);
                  syn_id_vector.push_back(syn_id);
                  num_edges++;
                }
            }

          edge_attr_values[0].insert(syn_id_vector);

          if (num_edges > 0)
            {
              if (edge_map.find(dst) == edge_map.end())
                {
                  edge_map.insert(make_pair(dst,make_tuple(adj_vector, edge_attr_values)));
                }
              else
                {
                  edge_tuple_t et = edge_map[dst];
                  vector<NODE_IDX_T> &v = get<0>(et);
                  vector <data::AttrVal> &a = get<1>(et);
                  v.insert(v.end(),adj_vector.begin(),adj_vector.end());
                  a[0].append(edge_attr_values[0]);
                  edge_map[dst] = make_tuple(v,a);
                }
            }
        }
    }


  return ierr;
}


int append_adj_map
(
 const vector<NODE_IDX_T>&   src_range,
 const int src_offset, const int dst_offset,
 const vector<NODE_IDX_T>&   dst_idx,
 const vector<DST_PTR_T>&    src_idx_ptr,
 const vector<NODE_IDX_T>&   src_idx,
 const map <string, data::AttrVal>& edge_attr_map,
 size_t&                     num_edges,
 edge_map_t&                 edge_map
 )
{
  int ierr = 0; 
  num_edges = 0;

  if (dst_idx.size() > 0)
    {
      for (size_t d = 0; d < dst_idx.size(); d++)
        {
          NODE_IDX_T dst = dst_idx[d] + dst_offset;

          size_t low_src_ptr = src_idx_ptr[d],
            high_src_ptr = src_idx_ptr[d+1];

          vector<NODE_IDX_T> adj_vector;
          vector <data::AttrVal> edge_attr_values_vector(edge_attr_map.size());

          {
            size_t ns_index=0;
            for (auto iter : edge_attr_map)
              {
                auto & edge_attrs = iter.second;
                auto & edge_attr_values = edge_attr_values_vector[ns_index];
                edge_attr_values.resize<float>(edge_attrs.size_attr_vec<float> ());
                edge_attr_values.resize<uint8_t>(edge_attrs.size_attr_vec<uint8_t> ());
                edge_attr_values.resize<uint16_t>(edge_attrs.size_attr_vec<uint16_t> ());
                edge_attr_values.resize<uint32_t>(edge_attrs.size_attr_vec<uint32_t> ());
                edge_attr_values.resize<int8_t>(edge_attrs.size_attr_vec<int8_t> ());
                edge_attr_values.resize<int16_t>(edge_attrs.size_attr_vec<int16_t> ());
                edge_attr_values.resize<int32_t>(edge_attrs.size_attr_vec<int32_t> ());
                ns_index++;
              }
          }
          
          for (size_t i = low_src_ptr; i < high_src_ptr; ++i)
            {
              NODE_IDX_T src = src_idx[i];
              if (src <= src_range[1] && src >= src_range[0])
                {
                  NODE_IDX_T src1 = src + src_offset;
                  adj_vector.push_back(src1);
                  size_t ns_index = 0;
                  for (auto iter : edge_attr_map)
                    {
                      auto & edge_attrs = iter.second;
                      auto & edge_attr_values = edge_attr_values_vector[ns_index];
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<float> (); ai++)
                        {
                          float v = edge_attrs.at<float>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<uint8_t> (); ai++)
                        {
                          uint8_t v = edge_attrs.at<uint8_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<uint16_t> (); ai++)
                        {
                          uint16_t v = edge_attrs.at<uint16_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<uint32_t> (); ai++)
                        {
                          uint32_t v = edge_attrs.at<uint32_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<int8_t> (); ai++)
                        {
                          int8_t v = edge_attrs.at<int8_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<int16_t> (); ai++)
                        {
                          int16_t v = edge_attrs.at<int16_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      for (size_t ai=0; ai<edge_attrs.size_attr_vec<int32_t> (); ai++)
                        {
                          int32_t v = edge_attrs.at<int32_t>(ai, i);
                          edge_attr_values.push_back(ai, v);
                        }
                      ns_index++;
                    }
                  num_edges++;
                }
            }

          if (num_edges > 0)
            {
              if (edge_map.find(dst) == edge_map.end())
                {
                  edge_map.insert(make_pair(dst,make_tuple(adj_vector, edge_attr_values_vector)));
                }
              else
                {
                  edge_tuple_t et = edge_map[dst];
                  vector<NODE_IDX_T> &v = get<0>(et);
                  vector <data::AttrVal> &va = get<1>(et);
                  v.insert(v.end(),adj_vector.begin(),adj_vector.end());
                  size_t ns_index = 0;
                  for (auto & a : va)
                    {
                      a.append(edge_attr_values_vector[ns_index]);
                      ns_index++;
                    }
                  edge_map[dst] = make_tuple(v,va);
                }
            }

        }
    }


  return ierr;
}



/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  int status=0;
  string dst_pop_name, src_pop_name;
  string output_file_name;
  string txt_filelist_file_name;
  vector <string> txt_input_file_names;
  string hdf5_input_file_name, hdf5_input_dsetpath;
  map < string, vector <size_t> > num_edge_attrs;
  map<string, vector< vector<string> > > edge_attr_names;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size, io_size=1;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  int dst_offset=0, src_offset=0;
  int optflag_attr_names   = 0;
  int optflag_io_size      = 0;
  int optflag_input_format = 0;
  int optflag_dst_offset   = 0;
  int optflag_src_offset   = 0;
  bool opt_attr_names = false;
  bool opt_io_size    = false;
  bool opt_txt        = false;
  bool opt_hdf5_syn   = false;
  bool opt_dst_offset = false,
    opt_src_offset    = false;

  // parse arguments
  static struct option long_options[] = {
    {"dst-offset",    required_argument, &optflag_dst_offset,  1 },
    {"src-offset",    required_argument, &optflag_src_offset,  1 },
    {"format",        required_argument, &optflag_input_format,  1 },
    {"io-size",       required_argument, &optflag_io_size,  1 },
    {"attributes",    required_argument, &optflag_attr_names,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "a:d:f:hi:s:", long_options, &option_index)) != -1)
    {
      switch (c)
        {
        case 0:
          if (optflag_attr_names == 1) {
            opt_attr_names = true;

            stringstream ss;
            string arg = string(optarg);
            string nsindex_delimiter = ":";
            string name_delimiter = ",";
            vector <string> attr_type_spec;
            tokenize(arg, nsindex_delimiter, attr_type_spec);

            string attr_namespace = attr_type_spec[0];
            string attr_index_str = attr_type_spec[1];
            string attr_names_str = attr_type_spec[2];
            size_t attr_index     = 0;
            
            ss << attr_index_str;
            ss >> attr_index;

            vector <string> attr_names;
            tokenize(attr_names_str, name_delimiter, attr_names);

            num_edge_attrs[attr_namespace].resize(data::AttrVal::num_attr_types);
            for (auto & attr_name : attr_names)
              {
                edge_attr_names[attr_namespace][attr_index].push_back(attr_name);
                num_edge_attrs[attr_namespace][attr_index]++;
              }

            optflag_attr_names=0;
          }
          if (optflag_io_size == 1) {
            stringstream ss;
            opt_io_size = true;
            ss << string(optarg);
            ss >> io_size;
            optflag_io_size=0;
          }
          if (optflag_dst_offset == 1) {
            stringstream ss;
            opt_dst_offset = true;
            ss << string(optarg);
            ss >> dst_offset;
            optflag_dst_offset=0;
          }
          if (optflag_src_offset == 1) {
            stringstream ss; 
            opt_src_offset = true;
            ss << string(optarg);
            ss >> src_offset;
            optflag_src_offset=0;
          }
          if (optflag_input_format == 1) {
            string input_format = string(optarg);
            if (input_format == "hdf5:syn")
              {
                opt_hdf5_syn = true;
              }
            if (input_format == "txt")
              {
                opt_txt = true;
              }
            optflag_input_format=0;
          }
          break;
        case 'f':
          {
            string input_format = string(optarg);
            if (input_format == "hdf5:syn")
              {
                opt_hdf5_syn = true;
              }
            if (input_format == "txt")
              {
                opt_txt = true;
              }
          }
          break;
        case 'd':
          {
            string arg = string(optarg);
            string delimiter = ":";
            vector <string> hdf5_spec;

            tokenize(arg, delimiter, hdf5_spec);

            hdf5_input_file_name = hdf5_spec[0];
            hdf5_input_dsetpath  = hdf5_spec[1];
          }
          break;
        case 'a':
          {
            opt_attr_names = true;

            stringstream ss;
            string arg = string(optarg);
            string nsindex_delimiter = ":";
            string name_delimiter = ",";
            vector <string> attr_type_spec;
            tokenize(arg, nsindex_delimiter, attr_type_spec);

            string attr_namespace = attr_type_spec[0];
            string attr_index_str = attr_type_spec[1];
            string attr_names_str = attr_type_spec[2];
            size_t attr_index     = 0;
            
            ss << attr_index_str;
            ss >> attr_index;

            vector <string> attr_names;
            tokenize(attr_names_str, name_delimiter, attr_names);
            num_edge_attrs[attr_namespace].resize(data::AttrVal::num_attr_types);
            for (auto & attr_name : attr_names)
              {
                edge_attr_names[attr_namespace][attr_index].push_back(attr_name);
                num_edge_attrs[attr_namespace][attr_index]++;
              }

            optflag_attr_names=0;
          }
          break;
        case 'i':
          {
            txt_filelist_file_name = string(optarg);
          }
          break;
        case 's':
          {
            stringstream ss;
            opt_io_size = true;
            ss << string(optarg);
            ss >> io_size;
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
      src_pop_name     = std::string(argv[optind]);
      dst_pop_name     = std::string(argv[optind+1]);
      output_file_name = std::string(argv[optind+2]);
      if (!opt_hdf5_syn && (!opt_txt))
        {
          print_usage_full(argv);
          exit(1);
        }
    }
  else
    {
      print_usage_full(argv);
      exit(1);
    }

  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
  vector<pair <pop_t, string> > pop_labels;
  size_t src_pop_idx, dst_pop_idx; bool src_pop_set=false, dst_pop_set=false;
  size_t n_nodes;
  vector<NODE_IDX_T>  src_range(2);
  
  vector<NODE_IDX_T>  dst_idx;
  vector<DST_PTR_T>   src_idx_ptr;
  vector<NODE_IDX_T>  src_idx;
  vector<DST_PTR_T>   syn_idx_ptr;
  vector<NODE_IDX_T>  syn_idx;

  if (opt_hdf5_syn)
    {
      assert(cell::read_population_ranges(all_comm, hdf5_input_file_name, pop_ranges,
                                          pop_vector, n_nodes) >= 0);
      assert(cell::read_population_labels(all_comm, hdf5_input_file_name, pop_labels) >= 0);
    }
  else
    {
      assert(cell::read_population_ranges(all_comm, output_file_name, pop_ranges,
                                          pop_vector, n_nodes) >= 0);
      assert(cell::read_population_labels(all_comm, output_file_name, pop_labels) >= 0);
    }

  for (size_t i=0; i< pop_labels.size(); i++)
    {
      if (src_pop_name == get<1>(pop_labels[i]))
        {
          src_pop_idx = get<0>(pop_labels[i]);
          src_pop_set = true;
        }
      if (dst_pop_name == get<1>(pop_labels[i]))
        {
          dst_pop_idx = get<0>(pop_labels[i]);
          dst_pop_set = true;
        }
    }
  assert(dst_pop_set && src_pop_set);
  
  src_range[0] = pop_vector[src_pop_idx].start;
  src_range[1] = src_range[0] + pop_vector[src_pop_idx].count;
  
  if (opt_hdf5_syn)
    {
      
      status = io::read_syn_projection (all_comm,
                                        hdf5_input_file_name,
                                        hdf5_input_dsetpath,
                                        dst_idx,
                                        src_idx_ptr,
                                        src_idx,
                                        syn_idx_ptr,
                                        syn_idx);

    }

  if (opt_txt)
        {
          ifstream infile(txt_filelist_file_name);
          string line;
          
          while (getline(infile, line))
            {
              stringstream ss;
              string file_name;
              ss << line;
              ss >> file_name;
              txt_input_file_names.push_back(file_name);
            }
        }
  
  map <string, data::AttrVal> edge_attrs;
  if (opt_txt)
    {
      // determine which connection files are read by which rank
      vector< pair<hsize_t,hsize_t> > ranges;
      mpi::rank_ranges(txt_input_file_names.size(), size, ranges);
      
      hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;

      for (size_t i=start; i<end; i++)
        {
          string txt_input_file_name = txt_input_file_names[i];
          
          status = io::read_txt_projection (txt_input_file_name, num_edge_attrs,
                                            dst_idx, src_idx_ptr, src_idx,
                                            edge_attrs);
        }
    }

  edge_map_t edge_map;
  size_t num_edges;

  /*
  model::NamedAttrMap edge_attr_map:
  edge_attr_map.uint32_values.resize(1);
  edge_attr_map.uint32_names.insert(make_pair("syn_id", 0));
  */

  if (syn_idx.size() > 0)
    {
      status = append_syn_adj_map (src_range, src_offset, dst_offset,
                                   dst_idx, src_idx_ptr, src_idx,
                                   syn_idx_ptr, syn_idx,
                                   num_edges, edge_map);
      edge_attr_names["Attributes"][data::AttrVal::attr_index_uint32].push_back("syn_id");
    }
  else
    {
      status = append_adj_map (src_range, src_offset, dst_offset,
                               dst_idx, src_idx_ptr, src_idx,
                               edge_attrs, num_edges, edge_map);
    }


  status = graph::write_graph (all_comm, io_size, output_file_name,
                               src_pop_name, dst_pop_name,
                               edge_attr_names, edge_map);

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
