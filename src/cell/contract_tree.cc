// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file contract_tree.cc
///
///  Tree contraction routine.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <cstdio>
#include <set>
#include <map>
#include <vector>
#include "ngraph.hh"

#include "neuroh5_types.hh"

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace NGraph;

namespace neuroh5
{
  namespace cell
  {
  
    void contract_tree_bfs (const Graph &A,
                            const vector<SWC_TYPE_T>& types, 
                            Graph::vertex_set& roots,
                            Graph &S, contraction_map_t& contraction_map,
                            Graph::vertex sp, Graph::vertex spp)
    {
    
      for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); ++p)
        {
          vector <Graph::vertex> section_members; 
          Graph::vertex s, v = Graph::node (p); 
          LAYER_IDX_T p_type = types[v];
                                     
          if (contraction_map[sp].size() > 1)
            {
              if (spp != v) section_members.push_back(spp);
              // creates an entry in the section map for this node
              section_members.push_back(v);
              // creates a new section node in the contraction map
              s = contraction_map.size();
              // insert new entry into contraction map
              // (key is section node, values is the set of nodes that are section members)
              contraction_map.insert(make_pair(s,section_members));
            
              // creates an entry in the section graph
              S.insert_vertex(s);
            
              // adds an edge from parent section to new section
              S.insert_edge(sp,s);
            }
          else
            {
              s = sp;
              contraction_map[s].push_back(v);
            }
        
          // identifies the node neighbors
          Graph::vertex_set outs = A.out_neighbors(v);
          bool type_change = false;
        
          while (outs.size() == 1)
            {
              if (types[v] != p_type)
                {
                  type_change = true;
                }
              else
                {
                  // adds the output node to the section map for the current section
                  v = Graph::node(outs.cbegin());
                  contraction_map[s].push_back(v);
                  // obtains the neighbors of the next node
                  outs = A.out_neighbors(v);
                }
            }
        
          // if the node is a branching point or region change point, recurse to create new section entry
          if ((outs.size() > 1) || type_change)
            {
              contract_tree_bfs(A, types, outs, S, contraction_map, s, v);
            }
        }

    }

    void contract_tree_dfs (const Graph &A, 
                            const vector<SWC_TYPE_T>& types, 
                            Graph::vertex_set& roots,
                            Graph &S, contraction_map_t& contraction_map,
                            Graph::vertex sp, Graph::vertex spp)
    {
      for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); ++p)
        {
          vector <Graph::vertex> section_members; 
          Graph::vertex s, v = Graph::node (p); 
          SWC_TYPE_T p_type = types[v];

          if (contraction_map[sp].size() > 1)
            {
              if (spp != v) section_members.push_back(spp);
              // creates an entry in the section map for this node
              section_members.push_back(v);
              // creates a new section node in the contraction map
              s = contraction_map.size();
              // insert new entry into contraction map
              // (key is section node, values is the set of nodes that are section members)
              contraction_map.insert(make_pair(s,section_members));
            
              // creates an entry in the section graph
              S.insert_vertex(s);
            
              // adds an edge from parent section to new section
              S.insert_edge(sp,s);
            }
          else
            {
              s = sp;
              contraction_map[s].push_back(v);
            }

          // identifies the node neighbors
          Graph::vertex_set outs = A.out_neighbors(v);
          bool type_change = false;
          
          while (outs.size() == 1)
            {
<<<<<<< HEAD
              if (types[v] != p_type)
                {
                  type_change = true;
                }
              else
                {
                  // adds the output node to the section map for the current section
                  v = Graph::node(outs.cbegin());
                  contraction_map[s].push_back(v);
                  // obtains the neighbors of the next node
                  outs = A.out_neighbors(v);
                }
=======
              Graph::vertex_set::const_iterator out = outs.cbegin();
              // adds the output node to the section map for the current section
              v = Graph::node(out);
              contraction_map[s].push_back(v);
              // obtains the neighbors of the next node
              outs = A.out_neighbors(v);
>>>>>>> 74b7e2a33ce019f08392c7f1a2ffd7af2c2d0dbb
            }

          // if the node is a branching point, recurse to create new section entry
          if ((outs.size() > 1) || type_change)
            {
              for ( Graph::vertex_set::const_iterator out = outs.cbegin(); out != outs.cend(); ++out)
                {
                  Graph::vertex_set new_root;
<<<<<<< HEAD
                  new_root.insert(*out);
                  contract_tree_dfs(A, types, new_root, S, contraction_map, s, v);
=======
                  new_root.insert(Graph::node(out));
                  contract_tree_dfs(A, new_root, S, contraction_map, s, v);
>>>>>>> 74b7e2a33ce019f08392c7f1a2ffd7af2c2d0dbb
                }
            }
        }

    }

    void contract_tree_regions_bfs (const Graph &A,
                                    const vector<LAYER_IDX_T>& regions,
                                    const vector<SWC_TYPE_T>& types, 
                                    Graph::vertex_set& roots,
                                    Graph &S, contraction_map_t& contraction_map,
                                    Graph::vertex sp, Graph::vertex spp)
    {
    
      for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); ++p)
        {
        
          vector <Graph::vertex> section_members; 
          Graph::vertex s, v = Graph::node (p); 
          LAYER_IDX_T p_region = regions[v];
          SWC_TYPE_T p_type = types[v];
                                     
          if (contraction_map[sp].size() > 1)
            {
              if (spp != v) section_members.push_back(spp);
              // creates an entry in the section map for this node
              section_members.push_back(v);
              // creates a new section node in the contraction map
              s = contraction_map.size();
              // insert new entry into contraction map
              // (key is section node, values is the set of nodes that are section members)
              contraction_map.insert(make_pair(s,section_members));
            
              // creates an entry in the section graph
              S.insert_vertex(s);
            
              // adds an edge from parent section to new section
              S.insert_edge(sp,s);
            }
          else
            {
              s = sp;
              contraction_map[s].push_back(v);
            }
        
          // identifies the node neighbors
          Graph::vertex_set outs = A.out_neighbors(v);
          bool region_change = false;
          bool type_change = false;
        
          while ((outs.size() == 1) && (!(region_change || type_change)))
            {
              // adds the output node to the section map for the current section
              v = Graph::node(outs.cbegin());
              if (regions[v] != p_region)
                {
                  region_change = true;
                }
              else if (types[v] != p_type)
                {
                  type_change = true;
                }
              else
                {
                  contraction_map[s].push_back(v);
                  // obtains the neighbors of the next node
                  outs = A.out_neighbors(v);
                }
            }
        
          // if the node is a branching point or region change point, recurse to create new section entry
          if ((outs.size() > 1) || region_change || type_change)
            {
              contract_tree_regions_bfs(A, regions, types, outs, S, contraction_map, s, v);
            }
        }

    }

    void contract_tree_regions_dfs (const Graph &A,
                                    const vector<LAYER_IDX_T>& regions,
                                    const vector<SWC_TYPE_T>& types, 
                                    Graph::vertex_set& roots,
                                    Graph &S, contraction_map_t& contraction_map,
                                    Graph::vertex sp, Graph::vertex spp)
    {
    
      for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); ++p)
        {
          vector <Graph::vertex> section_members; 
          Graph::vertex s, v = Graph::node (p); 
          LAYER_IDX_T p_region = regions[v];
          SWC_TYPE_T p_type = types[v];

          if (contraction_map[sp].size() > 1)
            {
              if (spp != v) section_members.push_back(spp);
              // creates an entry in the section map for this node
              section_members.push_back(v);
              // creates a new section node in the contraction map
              s = contraction_map.size();
              // insert new entry into contraction map
              // (key is section node, values is the set of nodes that are section members)
              contraction_map.insert(make_pair(s,section_members));
            
              // creates an entry in the section graph
              S.insert_vertex(s);
            
              // adds an edge from parent section to new section
              S.insert_edge(sp,s);
            }
          else
            {
              s = sp;
              contraction_map[s].push_back(v);
            }

          // identifies the node neighbors
          Graph::vertex_set outs = A.out_neighbors(v);
          bool region_change = false;
          bool type_change = false;
        
          while ((outs.size() == 1) && (!(region_change || type_change)))
            {
              // adds the output node to the section map for the current section
              v = Graph::node(outs.cbegin());
              if (regions[v] != p_region)
                {
                  region_change = true;
                }
              else if (types[v] != p_type)
                {
                  type_change = true;
                }
              else
                {
                  contraction_map[s].push_back(v);
                  // obtains the neighbors of the next node
                  outs = A.out_neighbors(v);
                }
            }

          // if the node is a branching point, recurse to create new section entry
          if ((outs.size() > 1) || region_change || type_change)
            {
              for ( Graph::vertex_set::const_iterator out = outs.begin(); out != outs.end(); ++out)
                {
                  Graph::vertex_set new_root;
                  new_root.insert(*out);
                  contract_tree_regions_dfs(A, regions, types, new_root, S, contraction_map, s, v);
                }
            }
        }

    }

  }
  
}
