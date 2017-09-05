// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file serialize_edge.cc
///
///  Top-level functions for serializing/deserializing graphs edges.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <vector>

#include "cereal/archives/portable_binary.hpp"

#undef NDEBUG
#include <cassert>

#include "neuroh5_types.hh"
using namespace neuroh5;


int main (int argc, char **argv)
{
  edge_map_t edge_map, edge_map_out;
  neuroh5::data::AttrVal attr_val;
  vector<NODE_IDX_T> adj_vector;
  adj_vector.push_back(0);
  adj_vector.push_back(1);
  adj_vector.push_back(2);
  
  edge_map.insert(make_pair(99, make_pair(adj_vector, attr_val)));
  std::stringstream ss;
          
  {
    cereal::PortableBinaryOutputArchive oarchive(ss); // Create an output archive
    oarchive(edge_map); // Write the data to the archive
    
  } // archive goes out of scope, ensuring all contents are flushed

  
  {
    cereal::PortableBinaryInputArchive iarchive(ss); // Create an input archive

    iarchive(edge_map_out); // Read the data from the archive
  }

  for (auto it = edge_map_out.begin(); it != edge_map_out.end(); ++it)
    {
      printf("%u %u\n", it->first, get<0>(it->second).size());
    }
  
}
