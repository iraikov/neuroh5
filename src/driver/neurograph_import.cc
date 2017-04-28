// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurograph_import.cc
///
///  Driver program for various import procedures.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "model_types.hh"
#include "population_reader.hh"
#include "projection_names.hh"
#include "read_syn_projection.hh"
#include "read_txt_projection.hh"
#include "write_graph.hh"
#include "attr_map.hh"
#include "edge_attr.hh"

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
  printf("Usage: %s  <SRC-POP> <DST-POP> <PRJ-NAME> <OUTPUT-FILE> \n\n", argv[0]);
  printf("Options:\n");
  printf("\t-i <FILE>:\n");
  printf("\t\tImport from given file\n");
  printf("\t-f <FORMAT>:\n");
  printf("\t\tInput format\n");

}

// Given a total number of elements and number of ranks, calculate the starting and length for each rank
void rank_ranges
(
 const size_t&                    num_elems,
 const size_t&                    size,
 vector< pair<hsize_t,hsize_t> >& ranges
 )
{
  hsize_t remainder=0, offset=0, buckets=0;
  ranges.resize(size);
  
  for (size_t i=0; i<size; i++)
    {
      remainder = num_elems - offset;
      buckets   = (size - i);
      ranges[i] = make_pair(offset, remainder / buckets);
      offset    += ranges[i].second;
    }
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
 model::edge_map_t&         edge_map
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
          model::EdgeAttr edge_attr_values;
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

          edge_attr_values.insert(syn_id_vector);

          if (edge_map.find(dst) == edge_map.end())
            {
              edge_map.insert(make_pair(dst,make_tuple(adj_vector, edge_attr_values)));
            }
          else
            {
              model::edge_tuple_t et = edge_map[dst];
              vector<NODE_IDX_T> &v = get<0>(et);
              model::EdgeAttr &a = get<1>(et);
              v.insert(v.end(),adj_vector.begin(),adj_vector.end());
              a.append(edge_attr_values);
              edge_map[dst] = make_tuple(v,a);
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
  string prj_name;
  string output_file_name;
  string txt_filelist_file_name;
  vector <string> txt_input_file_names;
  string hdf5_input_file_name, hdf5_input_dsetpath;
  vector <size_t> num_attrs;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size, io_size;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  int dst_offset=0, src_offset=0;
  int optflag_iosize = 0;
  int optflag_input_format = 0;
  int optflag_dst_offset = 0;
  int optflag_src_offset = 0;
  bool opt_iosize = false;
  bool opt_txt = false;
  bool opt_hdf5_syn = false;
  bool opt_dst_offset = false,
    opt_src_offset = false;

  // parse arguments
  static struct option long_options[] = {
    {"dst-offset",    required_argument, &optflag_dst_offset,  1 },
    {"src-offset",    required_argument, &optflag_src_offset,  1 },
    {"format",        required_argument, &optflag_input_format,  1 },
    {"iosize",        required_argument, &optflag_iosize,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hf:i:d:a:s:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
          if (optflag_iosize == 1) {
            opt_iosize = true;
            ss << string(optarg);
            ss >> io_size;
          }
          if (optflag_dst_offset == 1) {
            stringstream ss;
            opt_dst_offset = true;
            ss << string(optarg);
            ss >> dst_offset;
          }
          if (optflag_src_offset == 1) {
            stringstream ss;
            opt_src_offset = true;
            ss << string(optarg);
            ss >> src_offset;
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
            size_t pos = arg.find(delimiter);
            hdf5_input_file_name = arg.substr(0, pos); 
            hdf5_input_dsetpath = arg.substr(pos + delimiter.length(),
                                             arg.find(delimiter, pos + delimiter.length()));
          }
          break;
        case 'a':
          {
            string arg = string(optarg);
            string delimiter = ",";
            size_t pos=0, pos1;
            do
              {
                size_t nval;
                stringstream ss;
                pos1 = arg.find(delimiter, pos);
                ss << arg.substr(pos, pos1);
                ss >> nval;
                num_attrs.push_back(nval);
                if (pos != string::npos)
                  {
                    pos = pos1 + delimiter.length();
                  }
              } while (pos != string::npos);
          }
          break;
        case 'i':
          {
            txt_filelist_file_name = string(optarg);
          }
          break;
        case 's':
          {
            opt_iosize = true;
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

  if (optind < argc-3)
    {
      src_pop_name     = std::string(argv[optind]);
      dst_pop_name     = std::string(argv[optind+1]);
      prj_name         = std::string(argv[optind+2]);
      output_file_name = std::string(argv[optind+3]);
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

  vector<model::pop_range_t> pop_vector;
  map<NODE_IDX_T, pair<uint32_t,model::pop_t> > pop_ranges;
  vector<pair <model::pop_t, string> > pop_labels;
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
      
      assert(io::hdf5::read_population_ranges(all_comm, hdf5_input_file_name, pop_ranges,
                                              pop_vector, n_nodes) >= 0);
      assert(io::hdf5::read_population_labels(all_comm, hdf5_input_file_name, pop_labels) >= 0);
      
      status = io::hdf5::read_syn_projection (all_comm,
                                              hdf5_input_file_name,
                                              hdf5_input_dsetpath,
                                              dst_idx,
                                              src_idx_ptr,
                                              src_idx,
                                              syn_idx_ptr,
                                              syn_idx);

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
  
  vector<NODE_IDX_T>  dst, src;
  model::EdgeAttr edge_attrs;
  if (opt_txt)
    {
      // determine which connection files are read by which rank
      vector< pair<hsize_t,hsize_t> > ranges;
      rank_ranges(txt_input_file_names.size(), size, ranges);
      
      size_t filecount=0;
      hsize_t start=ranges[rank].first, end=ranges[rank].first+ranges[rank].second;

      for (size_t i=start; i<end; i++)
        {
          string txt_input_file_name = txt_input_file_names[i];
          
          status = io::read_txt_projection (txt_input_file_name, num_attrs,
                                            dst, src, edge_attrs);
        }
    }

  model::edge_map_t edge_map;
  size_t num_edges;

  /*
  model::NamedAttrMap edge_attr_map:
  edge_attr_map.uint32_values.resize(1);
  edge_attr_map.uint32_names.insert(make_pair("syn_id", 0));
  */
  
  status = append_syn_adj_map (src_range, src_offset, dst_offset,
                               dst_idx, src_idx_ptr, src_idx,
                               syn_idx_ptr, syn_idx,
                               num_edges, edge_map);

  vector<vector<string>> edge_attr_names(model::EdgeAttr::num_attr_types);
  edge_attr_names[model::EdgeAttr::attr_index_uint32].push_back("syn_id");
  status = graph::write_graph (all_comm, io_size, output_file_name, src_pop_name, dst_pop_name, prj_name, edge_attr_names, edge_map);

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
