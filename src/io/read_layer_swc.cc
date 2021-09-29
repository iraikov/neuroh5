// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_layer_swc.cc
///
///  Top-level functions for reading descriptions of neurite structures.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
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
#include <forward_list>

#include "ngraph.hh"
#include "neuroh5_types.hh"
#include "contract_tree.hh"
#include "throw_assert.hh"

using namespace std;
using namespace NGraph;

namespace neuroh5
{

  namespace io
  {
    
    /*****************************************************************************
     * Load tree data structures 
     *****************************************************************************/
    
    int read_layer_swc
    (
     const std::string& file_name,
     const CELL_IDX_T gid,
     const int id_offset,
     const int layer_offset,
     const SWC_TYPE_T swc_type,
     const bool split_layers,
     std::forward_list<neurotree_t> &tree_list
     )
    {
      int status = 0;
      Graph A, S;
      std::vector<COORD_T> xcoords, ycoords, zcoords;  // coordinates of nodes
      std::vector<REALVAL_T> radiuses;   // Radius
      std::vector<LAYER_IDX_T> layers;   // Layer
      std::vector<PARENT_NODE_IDX_T> parents;   // Parent point ids
      std::vector<SWC_TYPE_T> swc_types;   // SWC types
      Graph::vertex_set roots;

      ifstream infile(file_name);
      string line;
      size_t i = 0;
    
      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T id, idpar; int opt_idpar;
          int layer_value; LAYER_IDX_T layer;
          REALVAL_T radius;
          COORD_T x, y, z;

          iss >> id;
          id = id+id_offset;
          if (iss.fail()) continue;
        
          throw_assert_nomsg (iss >> layer_value);
          throw_assert_nomsg (iss >> x);
          throw_assert_nomsg (iss >> y);
          throw_assert_nomsg (iss >> z);
          throw_assert_nomsg (iss >> radius);
          throw_assert_nomsg (iss >> opt_idpar);

          if (layer_value < 0)
            {
              layer = 0;
            }
          else
            {
              layer = layer_value + layer_offset;
            }
        
          A.insert_vertex(id);
          if (opt_idpar > -1)
            {
              idpar = opt_idpar+id_offset;
              A.insert_edge(idpar,id);
            }
          else
            {
              roots.insert(id);
            }
        
          if (opt_idpar > -1)
            {
              parents.push_back(opt_idpar+id_offset);
            }
          else
            {
              parents.push_back(opt_idpar);
            }
          
          swc_types.push_back(swc_type);
          xcoords.push_back(x);
          ycoords.push_back(y);
          zcoords.push_back(z);
          radiuses.push_back(radius);
          layers.push_back(layer);
        
          i++;
        }
    
      infile.close();

      //cout << A;

      contraction_map_t contraction_map;
      vector<SECTION_IDX_T> src_vector, dst_vector;
      vector<SECTION_IDX_T> sec_vector;
      S.insert_vertex(0);

      if (split_layers)
        cell::contract_tree_regions_dfs (A, layers, swc_types, roots, S, contraction_map, 0, 0);
      else
        cell::contract_tree_dfs (A, swc_types, roots, S, contraction_map, 0, 0);
    
      //cout << S;
      size_t num_sections = contraction_map.size();
      throw_assert_nomsg(num_sections > 0);
    
      sec_vector.push_back(contraction_map.size());
      size_t sec_idx = 0;
      for(auto it = contraction_map.cbegin(); it != contraction_map.end(); it++)
        {
          // iterator->first = key
          // iterator->second = value
          size_t size = it->second.size();
          sec_vector.push_back(size);
          sec_vector.insert(std::end(sec_vector),std::begin(it->second),std::end(it->second));
          sec_idx++;
        }
      throw_assert_nomsg(sec_idx == num_sections);

      
      for ( Graph::const_iterator p = S.begin(); p != S.end(); p++)
        {
          Graph::vertex u = Graph::node (p); 
          Graph::vertex_set outs = S.out_neighbors(u);
          throw_assert_nomsg(u < num_sections);

          for ( Graph::vertex_set::const_iterator s = outs.begin(); s != outs.end(); s++)
            {
              Graph::vertex v = Graph::node (s);
              throw_assert_nomsg(v < num_sections);
              throw_assert_nomsg(u != v);
              src_vector.push_back(u);
              dst_vector.push_back(v);
            }
        }

    
      neurotree_t tree = make_tuple(gid,src_vector,dst_vector,sec_vector,xcoords,ycoords,zcoords,radiuses,layers,parents,swc_types);
      tree_list.push_front(tree);

      if (debug_enabled)
        {
          cout << "gid " << gid << ": " << endl;
          cout << "layers: " << endl;
          for_each(layers.cbegin(),
                   layers.cend(),
                   [] (const LAYER_IDX_T i)
                   { cout << " " << (unsigned int)i; } 
                   );
          cout << endl;

          cout << "src_vector: " << endl;
          for_each(src_vector.cbegin(),
                   src_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;

          cout << "dst_vector: " << endl;
          for_each(dst_vector.cbegin(),
                   dst_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;
        
          cout << "sec_vector: " << endl;
          for_each(sec_vector.cbegin(),
                   sec_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;
        }

      return status;
    }

    int read_swc
    (
     const std::string& file_name,
     const CELL_IDX_T gid,
     const int id_offset,
     std::forward_list<neurotree_t> &tree_list
     )
    {
      int status = 0;
      Graph A, S;
      std::vector<COORD_T> xcoords, ycoords, zcoords;  // coordinates of nodes
      std::vector<REALVAL_T> radiuses;   // Radius
      std::vector<LAYER_IDX_T> layers;   // Layer
      std::vector<PARENT_NODE_IDX_T> parents;   // Parent point ids
      std::vector<SWC_TYPE_T> swc_types;   // SWC types
      Graph::vertex_set roots;

      ifstream infile(file_name);
      string line;
      size_t i = 0;
    
      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T id, idpar; int opt_idpar;
          int swc_value; int opt_layer; LAYER_IDX_T layer=-1;
          SWC_TYPE_T swc_type;
          REALVAL_T radius;
          COORD_T x, y, z;

          iss >> id;
          id = id+id_offset;
          if (iss.fail()) continue;
        
          throw_assert_nomsg (iss >> swc_value);
          swc_type = swc_value;
          throw_assert_nomsg (iss >> x);
          throw_assert_nomsg (iss >> y);
          throw_assert_nomsg (iss >> z);
          throw_assert_nomsg (iss >> radius);
          throw_assert_nomsg (iss >> opt_idpar);
          iss >> opt_layer;
          if (!iss.fail())
            {
              layer = opt_layer;
            }
          
          
          A.insert_vertex(id);
          if (opt_idpar > -1)
            {
              idpar = opt_idpar+id_offset;
              A.insert_edge(idpar,id);
            }
          else
            {
              roots.insert(id);
            }
        
          if (opt_idpar > -1)
            {
              parents.push_back(opt_idpar+id_offset);
            }
          else
            {
              parents.push_back(opt_idpar);
            }
          
          swc_types.push_back(swc_type);
          xcoords.push_back(x);
          ycoords.push_back(y);
          zcoords.push_back(z);
          radiuses.push_back(radius);
          layers.push_back(layer);
        
          i++;
        }
    
      infile.close();

      //cout << A;
    
      contraction_map_t contraction_map;
      vector<SECTION_IDX_T> src_vector, dst_vector;
      vector<SECTION_IDX_T> sec_vector;
      S.insert_vertex(0);

      cell::contract_tree_dfs (A, swc_types, roots, S, contraction_map, 0, 0);
    
      //cout << S;
      size_t num_sections = contraction_map.size();
      throw_assert_nomsg(num_sections > 0);
    
      sec_vector.push_back(contraction_map.size());
      size_t sec_idx = 0;
      for(auto it = contraction_map.cbegin(); it != contraction_map.end(); it++)
        {
          // iterator->first = key
          // iterator->second = value
          size_t size = it->second.size();
          sec_vector.push_back(size);
          sec_vector.insert(std::end(sec_vector),std::begin(it->second),std::end(it->second));
          sec_idx++;
        }
      throw_assert_nomsg(sec_idx == num_sections);
    
      for ( Graph::const_iterator p = S.begin(); p != S.end(); p++)
        {
          Graph::vertex u = Graph::node (p); 
          Graph::vertex_set outs = S.out_neighbors(u);
          throw_assert_nomsg(u < num_sections);

          for ( Graph::vertex_set::const_iterator s = outs.begin(); s != outs.end(); s++)
            {
              Graph::vertex v = Graph::node (s);
              throw_assert_nomsg(v < num_sections);
              throw_assert_nomsg(u != v);
              src_vector.push_back(u);
              dst_vector.push_back(v);
            }
        }

    
      neurotree_t tree = make_tuple(gid,src_vector,dst_vector,sec_vector,xcoords,ycoords,zcoords,radiuses,layers,parents,swc_types);
      tree_list.push_front(tree);

      if (debug_enabled)
        {
          cout << "layers: " << endl;
          for_each(layers.cbegin(),
                   layers.cend(),
                   [] (const LAYER_IDX_T i)
                   { cout << " " << (int)i; } 
                   );
          cout << endl;

          cout << "SWC types: " << endl;
          for_each(swc_types.cbegin(),
                   swc_types.cend(),
                   [] (const SWC_TYPE_T i)
                   { cout << " " << (int)i; } 
                   );
          cout << endl;

          cout << "src_vector: " << endl;
          for_each(src_vector.cbegin(),
                   src_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;

          cout << "dst_vector: " << endl;
          for_each(dst_vector.cbegin(),
                   dst_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;
        
          cout << "sec_vector: " << endl;
          for_each(sec_vector.cbegin(),
                   sec_vector.cend(),
                   [] (const Graph::vertex i)
                   { cout << " " << i; } 
                   );
          cout << endl;
        }

      return status;
    }


  }
}
