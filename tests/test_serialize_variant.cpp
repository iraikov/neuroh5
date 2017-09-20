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

// type support
#include "cereal/types/vector.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/set.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/utility.hpp"

using namespace std;

#include "variant/variant.hpp"

#undef NDEBUG
#include <cassert>

#include "neuroh5_types.hh"
using namespace neuroh5;

struct Result {
  vector <int> data;
};

struct Error {
  int32_t code;
  vector<char> message;
};

using Response = mapbox::util::variant< Error, Result >;

struct ResponseWrapper {
  Response response;
};


template<class Archive>
void save(Archive & archive, ResponseWrapper const &m) 
{
  m.response.match([&archive] (Result r) { int tag=0; archive(tag, r.data); },
                   [&archive] (Error e)  { int tag=1; archive(tag, e.code, e.message); });
}

template<class Archive>
void load(Archive & archive, ResponseWrapper & m) 
{
  int tag;
  archive(tag);
  
  switch (tag)
    {
    case 0:
      {
        Result r;
        archive(r.data);
        m.response = r;
      }
      break;
    case 1:
      {
        Error e;
        archive(e.code, e.message);
        m.response = e;
      }
      break;
    }
}

int main (int argc, char **argv)
{
  map <int, pair< vector<NODE_IDX_T>, ResponseWrapper > >  data_map1, data_map2, data_map_out;
  vector <int> data(5, 1);

  Response response_data1 = Result { data };
  ResponseWrapper response1 = { response_data1 };
  vector<char> msg;
  msg.push_back('a');
  msg.push_back('b');
  msg.push_back('c');
  Response response_data2 = Error { 2, msg };
  ResponseWrapper response2 = { response_data2 };

  vector<NODE_IDX_T> adj_vector;
  adj_vector.push_back(0);
  adj_vector.push_back(1);
  adj_vector.push_back(2);
  vector<char> sendbuf;

  data_map1.insert(make_pair(99, make_pair(adj_vector, response1)));
  data_map2.insert(make_pair(99, make_pair(adj_vector, response2)));

  std::stringstream ss1, ss2;
          
  {
    cereal::PortableBinaryOutputArchive oarchive(ss1); // Create an output archive
    oarchive(data_map1); // Write the data to the archive
    
  } // archive goes out of scope, ensuring all contents are flushed

  copy(istream_iterator<char>(ss1), istream_iterator<char>(), back_inserter(sendbuf));

  {
    string s = string(sendbuf.begin(), sendbuf.end());
    stringstream ss_out(s);
    cereal::PortableBinaryInputArchive iarchive(ss_out); // Create an input archive

    iarchive(data_map_out); // Read the data from the archive
  }
  
  for (auto it = data_map_out.begin(); it != data_map_out.end(); ++it)
    {
      printf("%u %u\n", it->first, get<0>(it->second).size());
      get<1>(it->second).response.match ([] (Result r) { printf("Result\n"); },
                                         [] (Error e)  { printf("Error\n"); });
    }

  data_map_out.clear();
  sendbuf.clear();
  {
    cereal::PortableBinaryOutputArchive oarchive(ss2); // Create an output archive
    oarchive(data_map2); // Write the data to the archive
    
  } // archive goes out of scope, ensuring all contents are flushed

  copy(istream_iterator<char>(ss2), istream_iterator<char>(), back_inserter(sendbuf));
  printf("after sendbuf 2\n");

  {
    string s = string(sendbuf.begin(), sendbuf.end());
    stringstream ss_out(s);
    cereal::PortableBinaryInputArchive iarchive(ss_out); // Create an input archive

    iarchive(data_map_out); // Read the data from the archive
  }

  for (auto it = data_map_out.begin(); it != data_map_out.end(); ++it)
    {
      get<1>(it->second).response.match ([] (Result r) { printf("Result\n"); },
                                         [] (Error e)  { printf("Error\n"); });
    }
  
}
