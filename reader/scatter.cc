
#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "dbs_graph_reader.hh"
#include "population_reader.hh"
#include "edge_reader.hh"

#include "hdf5.h"

#include <getopt.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
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
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/


int append_edge_map
(
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const vector<float>&      longitudinal_distance,
 const vector<float>&      transverse_distance,
 const vector<float>&      distance,
 const vector<float>&      synaptic_weight,
 const vector<uint16_t>&   segment_index,
 const vector<uint16_t>&   segment_point_index,
 const vector<uint8_t>&    layer,
 const vector<rank_t>& node_rank_vector,
 const vector<uint8_t>& has_edge_attrs,
 size_t& num_edges,
 rank_edge_map_t & rank_edge_map
 )
{
  int ierr = 0; size_t dst_ptr_size;
  
  if (dst_blk_ptr.size() > 0) 
    {
      dst_ptr_size = dst_ptr.size();
      for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
        {
          size_t low_dst_ptr = dst_blk_ptr[b], high_dst_ptr = dst_blk_ptr[b+1];
          NODE_IDX_T dst_base = dst_idx[b];
          for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
            {
              if (i < dst_ptr_size-1) 
                {
                  NODE_IDX_T dst = dst_base + ii + dst_start;
                  rank_t dstrank = node_rank_vector[dst];
                  pair<rank_edge_map_iter_t,bool> r = rank_edge_map.insert(make_pair(dstrank, edge_map_t()));
                  pair<edge_map_iter_t,bool> n = rank_edge_map[dstrank].insert(make_pair(dst, edge_tuple_t()));

                  edge_tuple_t& et = rank_edge_map[dstrank][dst];

                  vector<NODE_IDX_T> &my_srcs = get<0>(et);
                  vector<float>&      my_longitudinal_distance = get<1>(et);
                  vector<float>&      my_transverse_distance   = get<2>(et);
                  vector<float>&      my_distance              = get<3>(et);
                  vector<float>&      my_synaptic_weight       = get<4>(et);
                  vector<uint16_t>&   my_segment_index         = get<5>(et);
                  vector<uint16_t>&   my_segment_point_index   = get<6>(et);
                  vector<uint8_t>&    my_layer                 = get<7>(et);

                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  for (size_t j = low; j < high; ++j)
                    {
                      NODE_IDX_T src = src_idx[j] + src_start;
                      my_srcs.push_back (src);
                      if (has_edge_attrs[0])
                        my_longitudinal_distance.push_back(longitudinal_distance[j]);
                      if (has_edge_attrs[1])
                        my_transverse_distance.push_back(transverse_distance[j]);
                      if (has_edge_attrs[2])
                        my_distance.push_back(distance[j]);
                      if (has_edge_attrs[3])
                        my_synaptic_weight.push_back(synaptic_weight[j]);
                      if (has_edge_attrs[4])
                        my_segment_index.push_back(segment_index[j]);
                      if (has_edge_attrs[5])
                        my_segment_point_index.push_back(segment_point_index[j]);
                      if (has_edge_attrs[6])
                        my_layer.push_back(layer[j]);
                      num_edges++;
                    }

                }
            }
        }
    }

  return ierr;
}

/*****************************************************************************
 * Read edge attributes
 *****************************************************************************/

int read_all_edge_attributes
(
 MPI_Comm comm,
 const char *input_file_name,
 const char *prj_name,
 const DST_PTR_T edge_base,
 const DST_PTR_T edge_count,
 const vector<string>&      edge_attr_names,
 vector<float>&      longitudinal_distance,
 vector<float>&      transverse_distance,
 vector<float>&      distance,
 vector<float>&      synaptic_weight,
 vector<uint16_t>&   segment_index,
 vector<uint16_t>&   segment_point_index,
 vector<uint8_t>&    layer
 )
{
  int ierr = 0; 
  vector<NODE_IDX_T> src_vec, dst_vec;

  for (size_t j = 0; j < edge_attr_names.size(); j++)
    {
      if (edge_attr_names[j].compare(string("Longitudinal Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Longitudinal Distance",
                                             edge_base, edge_count,
                                             LONG_DISTANCE_H5_NATIVE_T,
                                             longitudinal_distance) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Transverse Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Transverse Distance",
                                             edge_base, edge_count,
                                             TRANS_DISTANCE_H5_NATIVE_T,
                                             transverse_distance)  >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Distance",
                                             edge_base, edge_count,
                                             DISTANCE_H5_NATIVE_T,
                                             distance)  >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Segment Index")) == 0)
        {
          assert(read_edge_attributes<uint16_t>(comm,
                                                input_file_name,
                                                prj_name,
                                                "Segment Index",
                                                edge_base, edge_count,
                                                SEGMENT_INDEX_H5_NATIVE_T,
                                                segment_index) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Segment Point Index")) == 0)
        {
          assert(read_edge_attributes<uint16_t>(comm,
                                                input_file_name,
                                                prj_name,
                                                "Segment Point Index",
                                                edge_base, edge_count,
                                                SEGMENT_POINT_INDEX_H5_NATIVE_T,
                                                segment_point_index) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Layer")) == 0)
        {
          assert(read_edge_attributes<uint8_t>(comm,
                                               input_file_name,
                                               prj_name,
                                               "Layer",
                                               edge_base, edge_count,
                                               LAYER_H5_NATIVE_T,
                                               layer) >= 0);
          continue;
        }

    }
  return ierr;
}



/*****************************************************************************
 * Prepare an MPI packed data structure with source vertices and edge attributes
 * for a given destination vertex.
 *****************************************************************************/

int pack_edge
(
 MPI_Comm comm,
 const NODE_IDX_T &dst,
 const vector<NODE_IDX_T>& src_vect,
 const vector<float>&      longitudinal_distance_vect,
 const vector<float>&      transverse_distance_vect,
 const vector<float>&      distance_vect,
 const vector<float>&      synaptic_weight_vect,
 const vector<uint16_t>&   segment_index_vect,
 const vector<uint16_t>&   segment_point_index_vect,
 const vector<uint8_t>&    layer_vect,
 int &sendpos,
 int &sendsize,
 vector<uint8_t> &sendbuf
 )
{
  int ierr = 0;
  int packsize;

  sendsize = 0;
  
  assert(MPI_Pack_size(1, NODE_IDX_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(src_vect.size(), NODE_IDX_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(longitudinal_distance_vect.size(), DISTANCE_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(transverse_distance_vect.size(), DISTANCE_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(distance_vect.size(), DISTANCE_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(synaptic_weight_vect.size(), SYNAPTIC_WEIGHT_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(segment_index_vect.size(), SEGMENT_INDEX_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(segment_point_index_vect.size(), SEGMENT_POINT_INDEX_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;
  assert(MPI_Pack_size(layer_vect.size(), LAYER_MPI_T, comm, &packsize) == MPI_SUCCESS);
  sendsize += packsize;

  sendbuf.resize(sendbuf.size() + sendsize);

  
  size_t sendbuf_size = sendbuf.size();
  uint32_t dst_numitems = 0;

  // Create MPI_PACKED object with all the source vertices and edge attributes
  assert(MPI_Pack(&dst, 1, NODE_IDX_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm) >= 0);
  
  dst_numitems = src_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&src_vect[0], src_vect.size(), NODE_IDX_MPI_T,
           &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = longitudinal_distance_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&longitudinal_distance_vect[0], longitudinal_distance_vect.size(),
           DISTANCE_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);

  dst_numitems = transverse_distance_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&transverse_distance_vect[0], transverse_distance_vect.size(),
           DISTANCE_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = distance_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&distance_vect[0], distance_vect.size(),
           DISTANCE_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = synaptic_weight_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&synaptic_weight_vect[0], synaptic_weight_vect.size(),
           SYNAPTIC_WEIGHT_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = segment_index_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&segment_index_vect[0], segment_index_vect.size(),
           SEGMENT_INDEX_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = segment_point_index_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&segment_point_index_vect[0], segment_point_index_vect.size(),
           SEGMENT_POINT_INDEX_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  
  dst_numitems = layer_vect.size();
  MPI_Pack(&dst_numitems, 1, MPI_UINT32_T, &sendbuf[0], sendbuf_size, &sendpos, comm);
  MPI_Pack(&layer_vect[0], layer_vect.size(),
           LAYER_MPI_T, &sendbuf[0], sendbuf_size, &sendpos, comm);

  
  
  return ierr;
}


/*****************************************************************************
 * Unpack an MPI packed edge data structure into source vertices and edge attributes
 *****************************************************************************/

int unpack_edge
(
 MPI_Comm comm,
 NODE_IDX_T &dst,
 vector<NODE_IDX_T>& src_vect,
 vector<float>&      longitudinal_distance_vect,
 vector<float>&      transverse_distance_vect,
 vector<float>&      distance_vect,
 vector<float>&      synaptic_weight_vect,
 vector<uint16_t>&   segment_index_vect,
 vector<uint16_t>&   segment_point_index_vect,
 vector<uint8_t>&    layer_vect,
 int & recvpos,
 const vector<uint8_t> &recvbuf
 )
{
  int ierr = 0;
  uint32_t dst_numitems;
  int recvbuf_size = recvbuf.size();

  
  assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst, 1, NODE_IDX_MPI_T, comm) >= 0);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  src_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &src_vect[0], dst_numitems, NODE_IDX_MPI_T,
             comm);
  
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  longitudinal_distance_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &longitudinal_distance_vect[0], dst_numitems, DISTANCE_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  transverse_distance_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &transverse_distance_vect[0], dst_numitems, DISTANCE_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  distance_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &distance_vect[0], dst_numitems, DISTANCE_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  synaptic_weight_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &synaptic_weight_vect[0], dst_numitems, SYNAPTIC_WEIGHT_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  segment_index_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &segment_index_vect[0], dst_numitems, SEGMENT_INDEX_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  segment_point_index_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &segment_point_index_vect[0], dst_numitems, SEGMENT_POINT_INDEX_MPI_T,
             comm);

  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos, &dst_numitems, 1, MPI_UINT32_T, comm);
  layer_vect.resize(dst_numitems);
  MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
             &layer_vect[0], dst_numitems, LAYER_MPI_T,
             comm);
  
  
  return ierr;
}




/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  char *input_file_name, *output_file_name, *rank_file_name;
  // MPI Communicator for I/O ranks
  MPI_Comm io_comm, all_comm;
  // MPI group color value used for I/O ranks
  int io_color = 1;
  // A vector that maps nodes to compute ranks
  vector<rank_t> node_rank_vector;
  // The set of compute ranks for which the current I/O rank is responsible
  rank_edge_map_t rank_edge_map;
  set< pair<pop_t, pop_t> > pop_pairs;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  vector<string> prj_names;
  size_t prj_size = 0;
  
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


  MPI_Comm_dup(MPI_COMM_WORLD,&all_comm);

  // Am I an I/O rank?
  if (rank < io_size)
    {
      MPI_Comm_split(all_comm,io_color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
      
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

  
      // read the population info
      assert(read_population_combos(io_comm, input_file_name, pop_pairs) >= 0);
      assert(read_population_ranges(io_comm, input_file_name, pop_ranges, pop_vector) >= 0);
      assert(read_projection_names(io_comm, input_file_name, prj_names) >= 0);
      prj_size = prj_names.size();
    }
  else
    {
      MPI_Comm_split(all_comm,0,rank,&io_comm);
    }
  MPI_Barrier(all_comm);

  
  assert(MPI_Bcast(&prj_size, 1, MPI_UINT64_T, 0, all_comm) >= 0);
      
  // For each projection, I/O ranks read the edges and scatter
  for (size_t i = 0; i < prj_size; i++)
    {
      vector<uint8_t> sendbuf; int sendpos = 0;
      vector<int> sendcounts, sdispls, recvcounts, rdispls;
      vector<NODE_IDX_T> recv_edges, total_recv_edges;
      rank_edge_map_t prj_rank_edge_map;
      vector<uint8_t> has_edge_attrs;
      uint32_t has_edge_attrs_length;
      
      sendcounts.resize(size,0);
      sdispls.resize(size,0);
      recvcounts.resize(size,0);
      rdispls.resize(size,0);

      if (rank < io_size)
        {
          DST_BLK_PTR_T block_base;
          DST_PTR_T edge_base, edge_count;
          NODE_IDX_T dst_start, src_start;
          vector<DST_BLK_PTR_T> dst_blk_ptr;
          vector<NODE_IDX_T> dst_idx;
          vector<DST_PTR_T> dst_ptr;
          vector<NODE_IDX_T> src_idx;
          size_t num_edges = 0, total_prj_num_edges = 0;
          vector<string> edge_attr_names;
          vector<float> longitudinal_distance;
          vector<float> transverse_distance;
          vector<float> distance;
          vector<float> synaptic_weight;
          vector<uint16_t> segment_index;
          vector<uint16_t> segment_point_index;
          vector<uint8_t> layer;

	  DEBUG("scatter: reading projection ", i, "(", prj_names[i], ")");
          assert(read_dbs_projection(io_comm, input_file_name, prj_names[i].c_str(), 
                                     pop_vector, dst_start, src_start, total_prj_num_edges,
                                     block_base, edge_base, dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
	  DEBUG("scatter: validating projection ", i, "(", prj_names[i], ")");
          // validate the edges
          assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                    pop_ranges, pop_pairs) == true);


          if (opt_attrs)
            {
              edge_count = src_idx.size();
	      DEBUG("scatter: validating edge attribute names from projection ", i, "(", prj_names[i], ")");
              assert(read_edge_attribute_names(MPI_COMM_WORLD, input_file_name, prj_names[i].c_str(), edge_attr_names) >= 0);
	      DEBUG("scatter: validating edge attributes from projection ", i, "(", prj_names[i], ")");
              assert(read_all_edge_attributes(MPI_COMM_WORLD, input_file_name, prj_names[i].c_str(), edge_base, edge_count,
                                              edge_attr_names, longitudinal_distance, transverse_distance, distance,
                                              synaptic_weight, segment_index, segment_point_index, layer) >= 0);
            }

          has_edge_attrs.push_back(longitudinal_distance.size() > 0);
          has_edge_attrs.push_back(transverse_distance.size() > 0);
          has_edge_attrs.push_back(distance.size() > 0);
          has_edge_attrs.push_back(synaptic_weight.size() > 0);
          has_edge_attrs.push_back(segment_index.size() > 0);
          has_edge_attrs.push_back(segment_point_index.size() > 0);
          has_edge_attrs.push_back(layer.size() > 0);

          // append to the edge map
          assert(append_edge_map(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx,
                                 longitudinal_distance, transverse_distance, distance,
                                 synaptic_weight, segment_index, segment_point_index, layer,
                                 node_rank_vector, has_edge_attrs, num_edges, prj_rank_edge_map) >= 0);
      
          // ensure that all edges in the projection have been read and appended to edge_list
          assert(num_edges == src_idx.size());

          
          dst_blk_ptr.clear();
          dst_idx.clear();
          dst_ptr.clear();
          src_idx.clear();
          longitudinal_distance.clear();
          transverse_distance.clear();
          distance.clear();
          synaptic_weight.clear();
          segment_index.clear();
          segment_point_index.clear();
          layer.clear();

	  DEBUG("scatter: packing edge data from projection ", i, "(", prj_names[i], ")");

          for (auto it1 = prj_rank_edge_map.cbegin(); it1 != prj_rank_edge_map.cend(); ++it1)
            {
              uint32_t dst_rank;
              dst_rank = it1->first;
              sdispls[dst_rank] = sendpos;
              if (it1->second.size() > 0)
                {
                  for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                    {
                      int sendsize;
                      NODE_IDX_T dst = it2->first;

                      const vector<NODE_IDX_T>  src_vect                   = get<0>(it2->second);
                      const vector<float>&      longitudinal_distance_vect = get<1>(it2->second);
                      const vector<float>&      transverse_distance_vect   = get<2>(it2->second);
                      const vector<float>&      distance_vect              = get<3>(it2->second);
                      const vector<float>&      synaptic_weight_vect       = get<4>(it2->second);
                      const vector<uint16_t>&   segment_index_vect         = get<5>(it2->second);
                      const vector<uint16_t>&   segment_point_index_vect   = get<6>(it2->second);
                      const vector<uint8_t>&    layer_vect                 = get<7>(it2->second);

                      if (src_vect.size() > 0)
                        {
                          
                          assert(pack_edge(all_comm, dst,
                                           src_vect,
                                           longitudinal_distance_vect,
                                           transverse_distance_vect,
                                           distance_vect,
                                           synaptic_weight_vect,
                                           segment_index_vect,
                                           segment_point_index_vect,
                                           layer_vect,
                                           sendpos,
                                           sendsize,
                                           sendbuf
                                           ) == 0);
                          
                          sendcounts[dst_rank] += sendsize;
                        }
                    }
                }


            }

                    
        }

      MPI_Barrier(all_comm);

      DEBUG("scatter: broadcasting edge attributes length ", i, "(", prj_names[i], ")");
      has_edge_attrs_length = has_edge_attrs.size();
      assert(MPI_Bcast(&has_edge_attrs_length, 1, MPI_UINT32_T, 0, all_comm) >= 0);
      if (rank > 0)
        {
          has_edge_attrs.resize(has_edge_attrs_length);
        }
      assert(MPI_Bcast(&has_edge_attrs[0], has_edge_attrs_length, MPI_UINT8_T, 0, all_comm) >= 0);
      
      DEBUG("scatter: alltoallv with edge data ", i, "(", prj_names[i], ")");

      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
      //    and creates sendcounts and sdispls arrays

      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT,
                          all_comm) >= 0);

      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls

      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < size; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      int recvpos = 0;
      vector<uint8_t> recvbuf(recvbuf_size);

      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv

      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) >= 0);
      sendbuf.clear();

      if (opt_output)
	{
	  if (opt_binary)
	    {
	      if (recvbuf.size() > 0) 
		{
		  ofstream outfile;
		  stringstream outfilename;
		  outfilename << string(output_file_name) << "." << i << "." << rank << ".edges.bin";
		  outfile.open(outfilename.str(), ios::binary);
		  while ((unsigned int)recvpos < recvbuf.size()-1)
		    {
		      NODE_IDX_T dst; 
		      vector<NODE_IDX_T> src_vect;
		      vector<float>      longitudinal_distance_vect;
		      vector<float>      transverse_distance_vect;
		      vector<float>      distance_vect;
		      vector<float>      synaptic_weight_vect;
		      vector<uint16_t>   segment_index_vect;
		      vector<uint16_t>   segment_point_index_vect;
		      vector<uint8_t>    layer_vect;
		      
		      unpack_edge(all_comm, dst,
				  src_vect,
				  longitudinal_distance_vect,
				  transverse_distance_vect,
				  distance_vect,
				  synaptic_weight_vect,
				  segment_index_vect,
				  segment_point_index_vect,
				  layer_vect,
				  recvpos,
				  recvbuf
				  );
		      
		      for (size_t k = 0; k < src_vect.size(); k++)
			{
			  NODE_IDX_T src = src_vect[k];
			  outfile << src << dst;
			  if (has_edge_attrs[0])
			    outfile << longitudinal_distance_vect[k];
			  if (has_edge_attrs[1])
			    outfile << transverse_distance_vect[k];
			  if (has_edge_attrs[2])
			    outfile << distance_vect[k];
			  if (has_edge_attrs[3])
			    outfile << synaptic_weight_vect[k];
			  if (has_edge_attrs[4])
			    outfile << segment_index_vect[k];
			  if (has_edge_attrs[5])
			    outfile << segment_point_index_vect[k];
			  if (has_edge_attrs[6])
			    outfile << layer_vect[k];
			}
		    }
		  outfile.close();
		}
	      
	      recvbuf.clear();
	    }
	  else
	    {
	      if (recvbuf.size() > 0) 
		{
		  int recvpos = 0;
		  ofstream outfile;
		  stringstream outfilename;
		  outfilename << string(output_file_name) << "." << i << "." << rank << ".edges";
		  outfile.open(outfilename.str());
		  while ((unsigned int)recvpos < recvbuf.size()-1)
		    {
		      NODE_IDX_T dst; 
		      vector<NODE_IDX_T> src_vect;
		      vector<float>      longitudinal_distance_vect;
		      vector<float>      transverse_distance_vect;
		      vector<float>      distance_vect;
		      vector<float>      synaptic_weight_vect;
		      vector<uint16_t>   segment_index_vect;
		      vector<uint16_t>   segment_point_index_vect;
		      vector<uint8_t>    layer_vect;
		      
		      unpack_edge(all_comm, dst,
				  src_vect,
				  longitudinal_distance_vect,
				  transverse_distance_vect,
				  distance_vect,
				  synaptic_weight_vect,
				  segment_index_vect,
				  segment_point_index_vect,
				  layer_vect,
				  recvpos,
				  recvbuf
				  );
		      for (size_t k = 0; k < src_vect.size(); k++)
			{
			  NODE_IDX_T src = src_vect[k];
			  outfile << "    " << src << " " << dst;
			  if (has_edge_attrs[0])
			    outfile << " " << longitudinal_distance_vect[k];
			  if (has_edge_attrs[1])
			    outfile << " " << transverse_distance_vect[k];
			  if (has_edge_attrs[2])
			    outfile << " " << distance_vect[k];
			  if (has_edge_attrs[3])
			    outfile << " " << synaptic_weight_vect[k];
			  if (has_edge_attrs[4])
			    outfile << " " << segment_index_vect[k];
			  if (has_edge_attrs[5])
			    outfile << " " << segment_point_index_vect[k];
			  if (has_edge_attrs[6])
			    outfile << " " << layer_vect[k];
			  outfile << std::endl;
			}
		    }
		  outfile.close();
		}
	      
	      recvbuf.clear();
	    }
	}
    }

  MPI_Barrier(all_comm);
  if (rank < io_size)
    {
      MPI_Comm_free(&io_comm);
    }
  MPI_Comm_free(&all_comm);
  
  MPI_Finalize();
  return 0;
}
