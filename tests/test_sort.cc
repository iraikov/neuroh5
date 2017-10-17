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

#undef NDEBUG
#include <cassert>

#include "neuroh5_types.hh"
#include "sort_permutation.hh"

using namespace neuroh5;


int main (int argc, char **argv)
{
  auto compare_nodes = [](const NODE_IDX_T& a, const NODE_IDX_T& b) { return (a < b); };
  vector<NODE_IDX_T> adj_vector;

  vector<size_t> p = data::sort_permutation(adj_vector, compare_nodes);
  
}
