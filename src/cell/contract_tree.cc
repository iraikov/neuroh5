// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file contract_tree.cc
///
///  Tree contraction routine.
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================

#include "debug.hh"

#include <cstdio>
#include <set>
#include <map>
#include <vector>
#include "ngraph.hh"
#include "neurotrees_types.hh"

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace NGraph;

namespace neurotrees
{
  
  void contract_tree_bfs (const Graph &A,  
                          Graph::vertex_set& roots,
                          Graph &S, contraction_map_t& contraction_map,
                          Graph::vertex sp, Graph::vertex spp)
  {
    
    for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); p++)
      {
        
        vector <Graph::vertex> section_members; 
        Graph::vertex s, v = Graph::node (p); 
                                     
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
        
        while (outs.size() == 1)
          {
            // adds the output node to the section map for the current section
            v = Graph::node(outs.cbegin());
            contraction_map[s].push_back(v);
            // obtains the neighbors of the next node
            outs = A.out_neighbors(v);
          }
        
        // if the node is a branching point or region change point, recurse to create new section entry
        if (outs.size() > 1)
          {
            contract_tree_bfs(A, outs, S, contraction_map, s, v);
          }
      }

  }

  void contract_tree_dfs (const Graph &A, 
                          Graph::vertex_set& roots,
                          Graph &S, contraction_map_t& contraction_map,
                          Graph::vertex sp, Graph::vertex spp)
  {
    
    for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); p++)
      {
        vector <Graph::vertex> section_members; 
        Graph::vertex s, v = Graph::node (p); 

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
        
        while (outs.size() == 1)
          {
            // adds the output node to the section map for the current section
            v = Graph::node(outs.cbegin());
            contraction_map[s].push_back(v);
            // obtains the neighbors of the next node
            outs = A.out_neighbors(v);
          }

        // if the node is a branching point, recurse to create new section entry
        if (outs.size() > 1)
          {
            for ( Graph::vertex_set::const_iterator out = outs.begin(); out != outs.end(); out++)
              {
                Graph::vertex_set new_root;
                new_root.insert(*out);
                contract_tree_dfs(A, new_root, S, contraction_map, s, v);
              }
          }
      }

  }

  void contract_tree_regions_bfs (const Graph &A,  const vector<LAYER_IDX_T>& regions, 
                                  Graph::vertex_set& roots,
                                  Graph &S, contraction_map_t& contraction_map,
                                  Graph::vertex sp, Graph::vertex spp)
  {
    
    for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); p++)
      {
        
        vector <Graph::vertex> section_members; 
        Graph::vertex s, v = Graph::node (p); 
        LAYER_IDX_T p_region = regions[v];
                                     
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
        
        while ((outs.size() == 1) && (!region_change))
          {
            // adds the output node to the section map for the current section
            v = Graph::node(outs.cbegin());
            if (regions[v] != p_region)
            {
                region_change = true;
              }
            else
              {
                contraction_map[s].push_back(v);
                // obtains the neighbors of the next node
                outs = A.out_neighbors(v);
              }
          }
        
        // if the node is a branching point or region change point, recurse to create new section entry
        if ((outs.size() > 1) || region_change)
          {
            contract_tree_regions_bfs(A, regions, outs, S, contraction_map, s, v);
          }
      }

  }

  void contract_tree_regions_dfs (const Graph &A, const vector<LAYER_IDX_T>& regions, 
                                  Graph::vertex_set& roots,
                                  Graph &S, contraction_map_t& contraction_map,
                                  Graph::vertex sp, Graph::vertex spp)
  {
    
    for ( Graph::vertex_set::const_iterator p = roots.begin(); p != roots.end(); p++)
      {
        vector <Graph::vertex> section_members; 
        Graph::vertex s, v = Graph::node (p); 
        LAYER_IDX_T p_region = regions[v];

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
        
        while ((outs.size() == 1) && (!region_change))
          {
            // adds the output node to the section map for the current section
            v = Graph::node(outs.cbegin());
            if (regions[v] != p_region)
              {
                region_change = true;
              }
            else
              {
                contraction_map[s].push_back(v);
                // obtains the neighbors of the next node
                outs = A.out_neighbors(v);
              }
          }

        // if the node is a branching point, recurse to create new section entry
        if ((outs.size() > 1) || region_change)
          {
            for ( Graph::vertex_set::const_iterator out = outs.begin(); out != outs.end(); out++)
              {
                Graph::vertex_set new_root;
                new_root.insert(*out);
                contract_tree_regions_dfs(A, regions, new_root, S, contraction_map, s, v);
              }
          }
      }

  }


  
}
