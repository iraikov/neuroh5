// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file write_graph.cc
///
///  Top-level functions for writing graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================


#include "debug.hh"

#include "attr_map.hh"
#include "model_types.hh"
#include "population_reader.hh"
#include "read_population.hh"
#include "write_graph.hh"
#include "write_connectivity.hh"
#include "write_edge_attributes.hh"
#include "hdf5_path_names.hh"
#include "sort_permutation.hh"

#include <vector>

#undef NDEBUG
#include <cassert>

using namespace ngh5::model;
using namespace std;

namespace ngh5
{
  namespace graph
  {


    // Assign each node to a rank 
    void compute_node_rank_vector
    (
     size_t num_ranks,
     size_t num_nodes,
     vector< rank_t > &node_rank_vector
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
    
      node_rank_vector.resize(num_nodes);
      for (size_t i=0; i<num_ranks; i++)
        {
          remainder  = num_nodes - offset;
          buckets    = num_ranks - i;
          for (size_t j = 0; j < remainder / buckets; j++)
            {
              node_rank_vector[offset+j] = i;
            }
          offset    += remainder / buckets;
        }
    }

    
    int write_graph
    (
<<<<<<< HEAD
     MPI_Comm         all_comm,
     MPI_Comm         io_comm,
     const int        io_size,
     const string&    file_name,
     const string&    src_pop_name,
     const string&    dst_pop_name,
     const string&    prj_name,
     const bool       opt_attrs,
     const vector<NODE_IDX_T>  edges,
     const model::NamedAttrMap& edge_attrs
     )
    {
      assert(edges.size()%2 == 0);

      uint64_t num_edges = edges.size()/2;
      
      // read the population info
      set< pair<model::pop_t, model::pop_t> > pop_pairs;
      vector<model::pop_range_t> pop_vector;
      vector<pair <model::pop_t, string> > pop_labels;
      map<NODE_IDX_T,pair<uint32_t,model::pop_t> > pop_ranges;
      size_t src_pop_idx, dst_pop_idx; bool src_pop_set=false, dst_pop_set=false;
      size_t total_num_nodes;

      int size, rank;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
      
      //FIXME: assert(io::hdf5::read_population_combos(comm, file_name, pop_pairs) >= 0);
      assert(io::hdf5::read_population_ranges(comm, file_name,
                                              pop_ranges, pop_vector, total_num_nodes) >= 0);
      assert(io::hdf5::read_population_labels(comm, file_name, pop_labels) >= 0);
      
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
      
      size_t dst_start = pop_vector[dst_pop_idx].start;
      size_t dst_end = dst_start + pop_vector[dst_pop_idx].count;
      size_t total_num_nodes = dst_start - dst_end;

      map<NODE_IDX_T, vector<NODE_IDX_T> > adj_map;

      for (size_t i = 1; i < edges.size(); i += 2)
        {
          // all source/destination node IDs must be in range
          assert(dst_start <= edges[i] && edges[i] < dst_end);
          if (!(src_start <= edges[i-1] && edges[i-1] < src_end))
            {
              printf("src_start = %lu src_end = %lu\n", src_start, src_end);
              printf("edges[%d] = %lu\n", i-1, edges[i-1]);
            }
          
          assert(src_start <= edges[i-1] && edges[i-1] < src_end);

          vector<NODE_IDX_T>& adj_vector  = adj_map[edges[i] - dst_start];
          adj_vector.push_back(edges[i-1] - src_start);
        }
      
      // compute sort permutations for the source arrays
      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };
      vector<std::size_t> src_sort_permutations;
      NODE_IDX_T edge_offset=0;
      for (size_t i = 0; i < adj_map.size(); i++)
        {
          iter = adj_map.find(i);
          vector<size_t> p = sort_permutation(iter->second, compare_nodes);
          apply_permutation_in_place(iter->second, p);
          for (size_t j=0; j<p.size(); j++)
            {
              vector<size_t> p = sort_permutation(iter->second, compare_nodes);
              apply_permutation_in_place(iter->second, p);
              for (size_t j=0; j<p.size(); j++)
                {
                  p[j] += edge_offset;
                }
              edge_offset += p.size();
              src_sort_permutations.insert(src_sort_permutations.end(),p.begin(),p.end());
            }
        }
      compute_node_rank_vector(io_size, total_num_nodes, node_rank_vector);

      // A vector that maps nodes to compute ranks
      vector <rank_t> node_rank_vector;
              
      // construct a map where each set of edges are arranged by destination I/O rank
      
             
      // send buffer and structures for MPI Alltoall operation
      vector<uint8_t> sendbuf;
      vector<int> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);


      // Create an MPI datatype to describe the sizes of edge structures
      Size sizeval;
      MPI_Datatype size_type, size_struct_type;
      MPI_Datatype size_fld_types[1] = { MPI_UINT32_T };
      int size_blocklen[1] = { 1 };
      MPI_Aint size_disp[1];
      
      size_disp[0] = reinterpret_cast<const unsigned char*>(&sizeval.size) - 
        reinterpret_cast<const unsigned char*>(&sizeval);
      assert(MPI_Type_create_struct(1, size_blocklen, size_disp, size_fld_types, &size_struct_type) == MPI_SUCCESS);
      assert(MPI_Type_create_resized(size_struct_type, 0, sizeof(sizeval), &size_type) == MPI_SUCCESS);
      assert(MPI_Type_commit(&size_type) == MPI_SUCCESS);
      
      EdgeHeader header;
      MPI_Datatype header_type, header_struct_type;
      MPI_Datatype header_fld_types[2] = { NODE_IDX_MPI_T, MPI_UINT32_T };
      int header_blocklen[2] = { 1, 1 };
      MPI_Aint header_disp[2];
      
      header_disp[0] = reinterpret_cast<const unsigned char*>(&header.key) - 
        reinterpret_cast<const unsigned char*>(&header);
      header_disp[1] = reinterpret_cast<const unsigned char*>(&header.size) - 
        reinterpret_cast<const unsigned char*>(&header);
      assert(MPI_Type_create_struct(2, header_blocklen, header_disp, header_fld_types, &header_struct_type) == MPI_SUCCESS);
      assert(MPI_Type_create_resized(header_struct_type, 0, sizeof(header), &header_type) == MPI_SUCCESS);
      assert(MPI_Type_commit(&header_type) == MPI_SUCCESS);

      // Create MPI_PACKED object with the edges of vertices for the respective I/O rank
      size_t num_packed_edges = 0; int sendpos = 0;
             
      pack_rank_adj_map (all_comm, header_type, size_type,
                         rank_node_vector, adj_map, num_packed_edges,
                         sendpos, sendbuf);

             
             
      
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      assert(file >= 0);

      io::hdf5::write_connectivity (file, prj_name, src_pop_idx, dst_pop_idx,
                                    src_start, src_end, dst_start, dst_end,
                                    num_edges, adj_map);

      
      const vector< map<NODE_IDX_T, float > >& float_attrs = edge_attrs.attr_maps<float>();
      for (auto & elem: edge_attrs.float_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map<NODE_IDX_T, float >& value_map = float_attrs[k];
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
          vector<float> values;
          for (const auto& val: value_map)
            {
              float v = val.second;
              values.push_back(v);
            }
          assert(values.size() == src_sort_permutations.size());
          apply_permutation_in_place<float>(values, src_sort_permutations);
          io::hdf5::write_sparse_edge_attribute<float>(file, path, values);
        }

      const vector< map<NODE_IDX_T, uint8_t > >& uint8_attrs = edge_attrs.attr_maps<uint8_t>();
      for (auto & elem: edge_attrs.uint8_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map<NODE_IDX_T, uint8_t >& value_map = uint8_attrs[k];
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
          vector<uint8_t> values;
          for (const auto& val: value_map)
            {
              uint8_t v = val.second;
              values.push_back(v);
            }
          assert(values.size() == src_sort_permutations.size());
          apply_permutation_in_place<uint8_t>(values, src_sort_permutations);
          io::hdf5::write_sparse_edge_attribute<uint8_t>(file, path, values);
        }

      const vector< map<NODE_IDX_T, uint16_t > >& uint16_attrs = edge_attrs.attr_maps<uint16_t>();
      for (const auto& elem: edge_attrs.uint16_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map <NODE_IDX_T, uint16_t >& value_map = uint16_attrs[k];
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
          vector<uint16_t> values;
          for (auto& val: value_map)
            {
              uint16_t v = val.second;
              values.push_back(v);
            }
          assert(values.size() == src_sort_permutations.size());
          apply_permutation_in_place<uint16_t>(values, src_sort_permutations);
          io::hdf5::write_sparse_edge_attribute<uint16_t>(file, path, values);
        }

      const vector< map<NODE_IDX_T, uint32_t > >& uint32_attrs = edge_attrs.attr_maps<uint32_t>();
      for (const auto& elem: edge_attrs.uint32_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map <NODE_IDX_T, uint32_t >& value_map = uint32_attrs[k];
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
          vector<uint32_t> values;
          for (auto& val: value_map)
            {
              uint32_t v = val.second;
              values.push_back(v);
            }
          assert(values.size() == src_sort_permutations.size());
          apply_permutation_in_place<uint32_t>(values, src_sort_permutations);
          io::hdf5::write_sparse_edge_attribute<uint32_t>(file, path, values);
        }
      
      assert(H5Fclose(file) >= 0);
      assert(H5Pclose(fapl) >= 0);

      return 0;
    }
  }
}
