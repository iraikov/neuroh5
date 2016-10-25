
#include "debug.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "graph_reader.hh"
#include "attributes.hh"

#include "ngh5paths.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#define MAX_PRJ_NAME 1024
#define MAX_EDGE_ATTR_NAME 1024

using namespace std;

namespace ngh5
{

/*****************************************************************************
 * Append src/dst node indices to a vector of edges
 *****************************************************************************/

int append_prj_list
(
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const vector< pair<string,hid_t> >& edge_attr_info,
 const vector<EdgeAttr> &edge_attr_values,
 size_t&                   num_edges,
 vector<prj_tuple_t>&      prj_list
 )
{
  int ierr = 0; size_t dst_ptr_size;
  num_edges = 0;
  vector<NODE_IDX_T> src_vec, dst_vec;
  vector<EdgeAttr> edge_attr_vec;

  //TODO: initialize members of edge_attr_vec to appropriate types
  
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
                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  for (size_t j = low; j < high; ++j)
                    {
                      NODE_IDX_T src = src_idx[j] + src_start;
                      src_vec.push_back(src);
                      dst_vec.push_back(dst);
                      for (size_t k = 0; k < edge_attr_vec.size(); k++)
                        {
                          switch (edge_attr_values[j].tag_active_type)
                            {
                            case EdgeAttr::at_float:    edge_attr_vec[k].push_back<float>(edge_attr_values[k].at<float>(j)); break;
                            case EdgeAttr::at_uint8:    edge_attr_vec[k].push_back<uint8_t>(edge_attr_values[k].at<uint8_t>(j)); break;
                            case EdgeAttr::at_uint16:   edge_attr_vec[k].push_back<uint8_t>(edge_attr_values[k].at<uint8_t>(j)); break;
                            case EdgeAttr::at_uint32:   edge_attr_vec[k].push_back<uint8_t>(edge_attr_values[k].at<uint8_t>(j)); break;
                            case EdgeAttr::at_null:     break;
                            }
                        }
		      num_edges++;
                    }
                }
            }
        }
    }

  prj_list.push_back(make_tuple(src_vec, dst_vec, edge_attr_info, edge_attr_vec));

  return ierr;
}

  
/*****************************************************************************
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/
/*
int append_edge_map
(
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const edge_attrval_t &    edge_attr_values,
 const vector<rank_t>&     node_rank_vector,
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
                  vector<edge_attrval_t> &my_edge_attr_vals = get<1>(et);

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
*/
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
 const vector< pair<string,hid_t> >& edge_attr_info,
 vector<EdgeAttr> &edge_attr_values
 )
{
  int ierr = 0; 
  vector<NODE_IDX_T> src_vec, dst_vec;

  for (size_t j = 0; j < edge_attr_info.size(); j++)
    {
      EdgeAttr attr_val;
      string attr_name = edge_attr_info[j].first;
      hid_t  attr_h5type = edge_attr_info[j].second;
      assert ((ierr = read_edge_attributes(comm,input_file_name,prj_name,attr_name.c_str(),
                                           edge_base, edge_count, attr_h5type, attr_val)) >= 0);
      edge_attr_values.push_back(attr_val);
    }
  return ierr;
}



/*****************************************************************************
 * Prepare an MPI packed data structure with source vertices and edge attributes
 * for a given destination vertex.
 *****************************************************************************/
/*
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
*/

/*****************************************************************************
 * Unpack an MPI packed edge data structure into source vertices and edge attributes
 *****************************************************************************/
/*
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

*/
/*****************************************************************************
 * Load projection data structures 
 *****************************************************************************/

int read_graph
(
 MPI_Comm comm,
 const char *input_file_name,
 const bool opt_attrs,
 const vector<string> prj_names,
 vector<prj_tuple_t> &prj_list,
 size_t &local_num_edges,
 size_t &total_num_edges
 )
 {
   // read the population info
   set< pair<pop_t, pop_t> > pop_pairs;
   assert(read_population_combos(comm, input_file_name, pop_pairs) >= 0);
   
   vector<pop_range_t> pop_vector;
   map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
   assert(read_population_ranges(comm, input_file_name, pop_ranges, pop_vector) >= 0);
   
  // read the edges
  for (size_t i = 0; i < prj_names.size(); i++)
    {
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base, edge_count;
      NODE_IDX_T dst_start, src_start;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;
      vector< pair<string,hid_t> > edge_attr_info;
      vector<EdgeAttr> edge_attr_values;
      size_t local_prj_num_edges;
      size_t total_prj_num_edges;
      
      //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

      assert(read_dbs_projection(comm, input_file_name, prj_names[i].c_str(), 
                                 pop_vector, dst_start, src_start, total_prj_num_edges, block_base, edge_base,
                                 dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
      // validate the edges
      assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, pop_ranges, pop_pairs) == true);
      
      if (opt_attrs)
        {
          edge_count = src_idx.size();
          assert(get_edge_attributes(input_file_name, prj_names[i], edge_attr_info) >= 0);

          assert(read_all_edge_attributes(comm, input_file_name, prj_names[i].c_str(), edge_base, edge_count,
                                          edge_attr_info, edge_attr_values) >= 0);
        }

      // append to the vectors representing a projection (sources, destinations, edge attributes)
      assert(append_prj_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, 
                             edge_attr_info, edge_attr_values, local_prj_num_edges, prj_list) >= 0);


      // ensure that all edges in the projection have been read and appended to edge_list
      assert(local_prj_num_edges == src_idx.size());

      //printf("Task %d has read %lu edges in projection %lu (%s)\n",
      //       rank,  local_prj_num_edges, i, prj_names[i].c_str());

      total_num_edges = total_num_edges + total_prj_num_edges;
      local_num_edges = local_num_edges + local_prj_num_edges;

    }

  return 0;
 }
  

/*****************************************************************************
 * Load and scatter edge data structures 
 *****************************************************************************/
/*
int scatter_graph
(
 MPI_Comm all_comm,
 const char *input_file_name,
 const int io_size,
 const bool opt_attrs,
 const vector<string> prj_names,
  // A vector that maps nodes to compute ranks
 const vector<rank_t> node_rank_vector,
 vector < edge_map_t > & prj_vector,
 vector < vector <uint8_t> > & has_edge_attrs_vector
 )
{
  int ierr = 0;
   // MPI Communicator for I/O ranks
  MPI_Comm io_comm;
  // MPI group color value used for I/O ranks
  int io_color = 1;
  // The set of compute ranks for which the current I/O rank is responsible
  rank_edge_map_t rank_edge_map;
  set< pair<pop_t, pop_t> > pop_pairs;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  vector<string> prj_names;
  size_t prj_size = 0;
  
  int rank, size;
  assert(MPI_Comm_size(all_comm, &size) >= 0);
  assert(MPI_Comm_rank(all_comm, &rank) >= 0);

  // Am I an I/O rank?
  if (rank < io_size)
    {
      MPI_Comm_split(all_comm,io_color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
      
  
      // read the population info
      assert(read_population_combos(io_comm, input_file_name, pop_pairs) >= 0);
      assert(read_population_ranges(io_comm, input_file_name, pop_ranges, pop_vector) >= 0);
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
      edge_map_t prj_edge_map;
      
      has_edge_attrs.resize(0,0);
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
              assert(read_edge_attribute_names(io_comm, input_file_name, prj_names[i].c_str(), edge_attr_names) >= 0);
	      DEBUG("scatter: validating edge attributes from projection ", i, "(", prj_names[i], ")");
              assert(read_all_edge_attributes(io_comm, input_file_name, prj_names[i].c_str(), edge_base, edge_count,
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
	} // rank < io_size

      if (opt_attrs)
	{
	  has_edge_attrs_length = has_edge_attrs.size();
	  assert(MPI_Bcast(&has_edge_attrs_length, 1, MPI_UINT32_T, 0, all_comm) >= 0);
	  has_edge_attrs.resize(has_edge_attrs_length);
	  assert(MPI_Bcast(&has_edge_attrs[0], has_edge_attrs_length, MPI_UINT8_T, 0, all_comm) >= 0);
	}


      if (rank < io_size)
	{
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
                          
                          assert(pack_edge(all_comm,
                                           dst,
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

	  DEBUG("scatter: finished packing edge data from projection ", i, "(", prj_names[i], ")");
        }

      // 1. Each ALL_COMM rank sends an edge vector size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
      //    and creates sendcounts and sdispls arrays

      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT, &recvcounts[0], 1, MPI_INT, all_comm) >= 0);

      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls

      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < size; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      vector<uint8_t> recvbuf(recvbuf_size);
      int recvpos = 0;

      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) >= 0);

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
          
          unpack_edge(all_comm,
                      dst,
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

          prj_edge_map.insert(make_pair(dst,
                                        make_tuple(src_vect,
                                                   longitudinal_distance_vect,
                                                   transverse_distance_vect,
                                                   distance_vect,
                                                   synaptic_weight_vect,
                                                   segment_index_vect,
                                                   segment_point_index_vect,
                                                   layer_vect)));

        }

      has_edge_attrs_vector.push_back(has_edge_attrs);
      prj_vector.push_back(prj_edge_map);
    }

  MPI_Comm_free(&io_comm);
  return ierr;
}
*/
}
