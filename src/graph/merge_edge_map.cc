// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file merge_edge_map.cc
///
///  Merge edges from multiple projections into a single edge map.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    void merge_edge_map (const vector < edge_map_t > &prj_vector,
                         map<NODE_IDX_T, vector<NODE_IDX_T> > &edge_map)
    {
      for (size_t i = 0; i < prj_vector.size(); i++)
        {
          edge_map_t prj_edge_map = prj_vector[i];
          if (prj_edge_map.size() > 0)
            {
              for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
                {
                  NODE_IDX_T dst   = it->first;
                  edge_tuple_t& et = it->second;
                
                  const vector<NODE_IDX_T>& src_vector = get<0>(et);

                  if (edge_map.find(dst) == edge_map.end())
                    {
                      edge_map.insert(make_pair(dst,src_vector));
                    }
                  else
                    {
                      vector<NODE_IDX_T> &v = edge_map[dst];
                      v.insert(v.end(),src_vector.begin(),src_vector.end());
                      edge_map[dst] = v;
                    }
                }
            }
        }
    }
  }
}
