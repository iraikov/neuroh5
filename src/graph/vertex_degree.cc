// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file vertex_degree.cc
///
///  Calculate vertex (in/out)degree from an edge map.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "model_types.hh"

#include <mpi.h>

#undef NDEBUG
#include <cassert>

using namespace ngh5::model;
using namespace std;

namespace ngh5
{
  namespace graph
  {
    
    int vertex_degree (MPI_Comm comm,
                       const map<NODE_IDX_T, vector<NODE_IDX_T> > &edge_map,
                       map< NODE_IDX_T, size_t > &degree_map,
                       uint32_t global_max_degree,
                       uint32_t global_min_degree)
    {
      int status;
      uint32_t max_degree=0, min_degree=0;
      for (auto it = edge_map.begin(); it != edge_map.end(); it++)
        {
          NODE_IDX_T vertex = it->first;
          const vector<NODE_IDX_T>& adj_vector = it->second;
          size_t degree = adj_vector.size();
          
          degree_map.insert(make_pair(vertex,degree));

          if (max_degree<degree)
            {
              max_degree = degree;
            }
          if (min_degree>degree)
            {
              min_degree = degree;
            }
        }

      status = MPI_Allreduce(&max_degree, &global_max_degree, 1, MPI_UINT32_T, MPI_MAX, comm);
      assert(status >= 0);
      status = MPI_Allreduce(&min_degree, &global_min_degree, 1, MPI_UINT32_T, MPI_MIN, comm);
      assert(status >= 0);

      return status;
    }
  }
  
}
