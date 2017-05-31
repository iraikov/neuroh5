// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file density_sample.cc
///
///  Top-level functions for sampling points in neurite structures according to a given density.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include "ngraph.hh"
#include "neurotrees_types.hh"
#include "contract_tree.hh"

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace NGraph;

namespace neuroio
{

  namespace cell
  {
  
    // linear interpolation
    float lerp(float v0, float v1, float x)
    {
      return (1.0-x)*v0 + x*v1;
    }
  
    float frustum_area (float r1, float r2, float h)
    {
      return M_PI * (r1 + r2) * sqrt((r1 - r2)*(r_u - r_v) + h*h);
    }
    
    int density_sample
    (
     const vector<neurotree_t> &tree_list,
     const map<LAYER_IDX_T, float> layer_density_map,
     const float sample_delta
     )
    {
      int status = 0;

      for (size_t i = 0; i < tree_list.size(); i++)
        {
          const neurotree_t &tree = tree_list[i];

          const CELL_IDX_T &gid = get<0>(tree);
          const std::vector<SECTION_IDX_T> & src_vector=get<1>(tree);
          const std::vector<SECTION_IDX_T> & dst_vector=get<2>(tree);
          const std::vector<SECTION_IDX_T> & sections=get<3>(tree);
          const std::vector<COORD_T> & xcoords=get<4>(tree);
          const std::vector<COORD_T> & ycoords=get<5>(tree);
          const std::vector<COORD_T> & zcoords=get<6>(tree);
          const std::vector<REALVAL_T> & radiuses=get<7>(tree);
          const std::vector<LAYER_IDX_T> & layers=get<8>(tree);
          const std::vector<PARENT_NODE_IDX_T> & parents=get<9>(tree);
          const std::vector<SWC_TYPE_T> & swc_types=get<10>(tree);

          Graph A, S;
          Graph::vertex_set roots;
        
          // Initialize point connectivity 
          NODE_IDX_T num_id = xcoords.size();
          for (NODE_IDX_T id=0; id<num_id; id++)
            {
              NODE_IDX_T idpar;
              PARENT_NODE_IDX_T opt_idpar = parents[id];
              LAYER_IDX_T layer = layers[id];
              REALVAL_T radius  = radiuses[id];
              COORD_T x = xcoords[id], y = ycoords[id], z = zcoords[id];
            
              A.insert_vertex(id);
              if (opt_idpar > -1)
                {
                  idpar = opt_idpar;
                  A.insert_edge(idpar,id);
                }
              else
                {
                  roots.insert(id);
                }
            
            }

          // Traverse the point graph and determine densities
          for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); p++)
            {
              Graph::vertex u = Graph::node (p); 
              const COORD_T x_u = xcoords[u], y_v = ycoords[u], z_v = zcoords[u];
              const REALVAL_t r_u = radiuses[u];
            
              // identifies the vertex neighbors
              Graph::vertex_set outs = A.out_neighbors(v);
            
              while (outs.size() == 1)
                {
                  Graph::vertex v = Graph::node(outs.cbegin());
                
                  const COORD_T x_v = xcoords[v], y_u = ycoords[v], z_u = zcoords[v];
                  const REALVAL_t r_v = radiuses[v];
                  const REALVAL_T h_uv = sqrt((x_v-x_u)*(x_v-x_u) + (y_v-y_u)*(y_v-y_u) + (z_v-z_u)*(z_v-z_u));
                  const vector<COORD_T> uv_vector({x_v-x_u,y_v-y_u,z_v-z_u});
                
                  vector<COORD_T> uprime({x_u, y_u, z_u});
                  REALVAL r_uprime = radiuses[u];
                  REALVAL_T h_uv_delta = (h_uv > sample_delta) ? sample_delta : h_uv;
                  REALVAL_T h_uv_prime = h_uv_delta;
                  while (h_uv_prime < h_uv)
                    {
                      vector<COORD_T> vprime(3);
                      mul_scalar_vector(h_uv_prime, uv_vector, vprime);
                      add_vector(u, vprime, vprime);
                      REALVAL_T loc  = h_uv_prime / h_uv;
                      REALVAL_T r_vprime = lerp(r_uprime, r_vprime, loc);
                      REALVAL_T area = frustum_area (r_uprime, r_vprime, h_uv_delta);
                      // TODO: determine density based on current layer
                      REALVAL_T num_samples = density * area;
                      // TODO: determine current section and location within section
                      // sloc = ...
                      section_sample_map.insert(s, make_tuple(sloc, num_samples);
                    
                                                h_uv_prime += h_uv_delta;
                                                uprime = vprime;
                                                r_uprime = r_vprime;
                                                }
                
                        // obtains the neighbors of the next node
                        outs = A.out_neighbors(u);
                    }
            
                  // if the node is a branching point, recurse to create new section entry
                  if (outs.size() > 1)
                    {
                      density_sample(A, outs, density_map, v);
                    }
                }
        
            }

          //cout << A;
    
          return status;
        }
  
    }
  }
