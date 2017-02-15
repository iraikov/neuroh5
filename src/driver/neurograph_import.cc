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


int append_edge_list
(
 const int src_offset, const int dst_offset,
 const vector<NODE_IDX_T>&  dst_idx,
 const vector<DST_PTR_T>&   src_idx_ptr,
 const vector<NODE_IDX_T>&  src_idx,
 const vector<DST_PTR_T>&   syn_idx_ptr,
 const vector<NODE_IDX_T>&  syn_idx,
 size_t&                    num_edges,
 vector<NODE_IDX_T>&        edge_list,
 model::NamedAttrMap&       edge_attr_map
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

          for (size_t i = low_src_ptr, ii = low_syn_ptr; i < high_src_ptr; ++i, ++ii)
            {
              assert(ii < high_syn_ptr);
              NODE_IDX_T src = src_idx[i] + src_offset;
              NODE_IDX_T syn_id = syn_idx[ii];
              edge_list.push_back(src);
              edge_list.push_back(dst);
              edge_attr_map.insert<uint32_t>(0, num_edges, syn_id);
              num_edges++;
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
  std::string dst_pop_name, src_pop_name;
  std::string prj_name;
  std::string output_file_name;
  std::string txt_input_filename;
  std::string hdf5_input_filename, hdf5_input_dsetpath;
  MPI_Comm all_comm;
  
  assert(MPI_Init(&argc, &argv) >= 0);

  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  int dst_offset=0, src_offset=0;
  bool opt_txt = true;
  bool opt_hdf5_syn = true;
  int optflag_input_format = 0;
  int optflag_dst_offset = 0;
  int optflag_src_offset = 0;
  bool opt_dst_offset = false,
    opt_src_offset = false;

  // parse arguments
  static struct option long_options[] = {
    {"dst-offset",    required_argument, &optflag_dst_offset,  1 },
    {"src-offset",    required_argument, &optflag_src_offset,  1 },
    {"format",        required_argument, &optflag_input_format,  1 },
    {0,         0,                 0,  0 }
  };
  char c;
  int option_index = 0;
  while ((c = getopt_long (argc, argv, "hf:i:d:", long_options, &option_index)) != -1)
    {
      stringstream ss;
      switch (c)
        {
        case 0:
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
            hdf5_input_filename = arg.substr(0, pos); 
            hdf5_input_dsetpath = arg.substr(pos + delimiter.length(),
                                             arg.find(delimiter, pos + delimiter.length()));
          }
          break;
        case 'i':
          {
            txt_input_filename = string(optarg);
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
      if (!opt_hdf5_syn || (!opt_txt))
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

  vector<NODE_IDX_T>  dst_idx;
  vector<DST_PTR_T>   src_idx_ptr;
  vector<NODE_IDX_T>  src_idx;
  vector<DST_PTR_T>   syn_idx_ptr;
  vector<NODE_IDX_T>  syn_idx;

  if (opt_hdf5_syn)
    status = io::hdf5::read_syn_projection (all_comm,
                                            hdf5_input_filename,
                                            hdf5_input_dsetpath,
                                            dst_idx,
                                            src_idx_ptr,
                                            src_idx,
                                            syn_idx_ptr,
                                            syn_idx);
  vector<NODE_IDX_T>  dst, src;
  model::EdgeAttr edge_attrs;
  vector <size_t> num_attrs;
  if (opt_txt)
    status = io::read_txt_projection (txt_input_filename, num_attrs,
                                      dst, src, edge_attrs);
  
  vector<NODE_IDX_T>  edges;
  size_t num_edges;
  model::NamedAttrMap edge_attr_map;
  edge_attr_map.uint32_values.resize(1);
  edge_attr_map.uint32_names.insert(make_pair("syn_id", 0));
  
  status = append_edge_list (src_offset, dst_offset,
                             dst_idx, src_idx_ptr, src_idx,
                             syn_idx_ptr, syn_idx,
                             num_edges,
                             edges,
                             edge_attr_map);

  status = graph::write_graph (all_comm, output_file_name, src_pop_name, dst_pop_name, prj_name, true, edges, edge_attr_map);

  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  
  return status;
}
