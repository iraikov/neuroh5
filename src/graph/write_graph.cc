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
#include "population_reader.hh"
#include "read_population.hh"
#include "write_graph.hh"
#include "write_connectivity.hh"
#include "write_edge_attributes.hh"
#include "hdf5_path_names.hh"
#include "sort_permutation.hh"

#undef NDEBUG
#include <cassert>

using namespace ngh5::model;
using namespace std;

namespace ngh5
{
  namespace graph
  {
    int write_graph
    (
     MPI_Comm              comm,
     const string&    file_name,
     const string&    src_pop_name,
     const string&    dst_pop_name,
     const string&    prj_name,
     const bool            opt_attrs,
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
      
      size_t src_start = pop_vector[src_pop_idx].start;
      size_t src_end = src_start + pop_vector[src_pop_idx].count;
      
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      assert(file >= 0);

      map<NODE_IDX_T, vector<NODE_IDX_T> > dst_src_map;
        
      for (NODE_IDX_T inode = dst_start; inode < dst_end; inode++)
        {
          dst_src_map.insert(make_pair(inode-dst_start, vector<NODE_IDX_T>()));
        }

      map<NODE_IDX_T, vector<NODE_IDX_T> >::iterator iter;
      for (size_t i = 1; i < edges.size(); i += 2)
        {
          // all source/destination node IDs must be in range
          assert(dst_start <= edges[i] && edges[i] < dst_end);
          assert(src_start <= edges[i-1] && edges[i-1] < src_end);
          
          iter = dst_src_map.find(edges[i] - dst_start);
          assert (iter != dst_src_map.end());
          iter->second.push_back(edges[i-1] - src_start);
        }

      // compute sort permutations for the source arrays
      auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };
      vector<vector<size_t>> src_sort_permutations;
      for (size_t i = 0; i < dst_src_map.size(); i++)
        {
          iter = dst_src_map.find(i);
          const vector<size_t> p = sort_permutation(iter->second, compare_nodes);
          apply_permutation_in_place(iter->second, p);
          src_sort_permutations.push_back(p);
        }
      
      io::hdf5::write_connectivity (file, prj_name, src_pop_idx, dst_pop_idx,
                                    src_start, src_end, dst_start, dst_end,
                                    num_edges, dst_src_map);

      dst_src_map.clear();
      const vector< map<NODE_IDX_T, float > >& float_attrs = edge_attrs.attr_maps<float>();
      for (auto & elem: edge_attrs.float_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map<NODE_IDX_T, float >& value_map = float_attrs[k];
          vector<float> values;
          for (const auto& val: value_map)
            {
              vector<float> v(val.second);
              const vector<size_t>& p = src_sort_permutations[val.first];
              apply_permutation_in_place(v, p);
              values.insert(values.end(),v.begin(),v.end());
            }
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
          io::hdf5::write_sparse_edge_attribute<float>(file, path, values);
        }

      const vector< map<NODE_IDX_T, uint8_t > >& uint8_attrs = edge_attrs.attr_maps<uint8_t>();
      for (auto & elem: edge_attrs.uint8_names)
        {
          const string& attr_name = elem.first;
          const size_t k = elem.second;
          const map<NODE_IDX_T, uint8_t >& value_map = uint8_attrs[k];
          vector<uint8_t> values;
          for (const auto& val: value_map)
            {
              vector<uint8_t> v(val.second);
              const vector<size_t>& p = src_sort_permutations[val.first];
              apply_permutation_in_place(v, p);
              values.insert(values.end(),v.begin(),v.end());
            }
          string path = io::hdf5::edge_attribute_path(prj_name, attr_name);
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
              vector<uint16_t> v(val.second);
              const vector<size_t>& p = src_sort_permutations[val.first];
              apply_permutation_in_place(v, p);
              values.insert(values.end(),v.begin(),v.end());
            }
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
              vector<uint32_t> v(val.second);
              const vector<size_t>& p = src_sort_permutations[val.first];
              apply_permutation_in_place(v, p);
              values.insert(values.end(),v.begin(),v.end());
            }
          io::hdf5::write_sparse_edge_attribute<uint32_t>(file, path, values);
        }
      
      assert(H5Fclose(file) >= 0);
      assert(H5Pclose(fapl) >= 0);

      return 0;
    }
  }
}
