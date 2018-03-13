// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_degree.cc
///
///  Calculate vertex (in/out)degree from an edge map.
///
///  Copyright (C) 2016-2018 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include <map>
#include <set>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace neuroh5;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    int vertex_degree (const vector < edge_map_t >& prj_vector,
                       const bool unique,
                       vector < map< NODE_IDX_T, size_t > > &degree_maps)
    {
      int status=0; 

      for (const map<NODE_IDX_T, edge_tuple_t >& edge_map : prj_vector)
        {
          vector <size_t> degree_vector;
          vector <NODE_IDX_T> node_id_vector;

          for (auto it = edge_map.begin(); it != edge_map.end(); ++it)
            {
              node_id_vector.push_back(it->first);
              const vector<NODE_IDX_T>& adj_vector = get<0>(it->second);
              size_t degree = 0;

              if (unique)
                {
                  set<NODE_IDX_T> s;
                  for (size_t i = 0; i < adj_vector.size(); ++i )
                    { s.insert( adj_vector[i] ); }
                  degree = s.size();
                }
              else
                {
                  degree = adj_vector.size();
                }
              degree_vector.push_back(degree);
            }
      
          map< NODE_IDX_T, size_t > degree_map;
          for (size_t i = 0; i < node_id_vector.size(); i++)
            {
              degree_map.insert(make_pair(node_id_vector[i], degree_vector[i]));
            }

          degree_maps.push_back(degree_map);
        }

      return status;
    }
  }
  
}
