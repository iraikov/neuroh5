// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file serialize_data.cc
///
///  Top-level functions for serializing/deserializing data objects.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#ifndef SERIALIZE_DATA_HH
#define SERIALIZE_DATA_HH

#include "debug.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <set>
#include <bitset>
#include <climits>
#include <map>
#include <vector>


// type support
#include "cereal/types/deque.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/utility.hpp"

#include "cereal/archives/binary.hpp"

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  namespace data
  {

    template<class T>
    void serialize_data (const T& data, 
                         vector<char>& sendbuf)
    {
      std::stringstream ss(ios::in | ios::out | ios::binary);
      {
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
        oarchive(data); // Write the data to the archive
        
      } // archive goes out of scope, ensuring all contents are flushed
      
      const string& sstr = ss.str();
      copy(sstr.begin(), sstr.end(), back_inserter(sendbuf));

    }

    
    template<class T>
    void deserialize_data (const vector<char> &recvbuf,
                           T& data)
    {
      {
        const string& s = string(recvbuf.begin(), recvbuf.end());
        stringstream ss(s, ios::in | ios::out | ios::binary);
        
        cereal::BinaryInputArchive iarchive(ss); // Create an input archive
        
        iarchive(data); // Read the data from the archive
      }
    }

  }
}
#endif
