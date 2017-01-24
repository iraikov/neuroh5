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

#include "edge_attr.hh"
#include "population_reader.hh"
#include "read_population.hh"
#include "write_graph.hh"
#include "write_connectivity.hh"

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
     const std::string&    file_name,
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    prj_name,
     const bool            opt_attrs,
     const vector<NODE_IDX_T>  edges,
     const model::EdgeNamedAttr& edge_attr_values
     )
    {

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

      io::hdf5::write_connectivity (file, prj_name, src_pop_idx, dst_pop_idx,
                                    src_start, src_end, dst_start, dst_end, edges);

      assert(H5Fclose(file) >= 0);
      assert(H5Pclose(fapl) >= 0);

      return 0;
    }
  }
}
