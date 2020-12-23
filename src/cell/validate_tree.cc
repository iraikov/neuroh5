// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file validate_tree.cc
///
///  Validate tree structure.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include <set>

#include "neuroh5_types.hh"
#include "throw_assert.hh"
#include "ngraph.hh"


using namespace std;
namespace neuroh5
{

  namespace cell
  {
    
    void validate_tree(const neurotree_t& tree)
    {
      const CELL_IDX_T tree_id = get<0>(tree);
      const std::vector<SECTION_IDX_T> & src_vector=get<1>(tree);
      const std::vector<SECTION_IDX_T> & dst_vector=get<2>(tree);
      const std::vector<SECTION_IDX_T> & sections=get<3>(tree);
      const std::vector<COORD_T> & xcoords=get<4>(tree);
      const std::vector<COORD_T> & ycoords=get<5>(tree);
      const std::vector<COORD_T> & zcoords=get<6>(tree);
      const std::vector<REALVAL_T> & radiuses=get<7>(tree);
      const std::vector<LAYER_IDX_T> & layers=get<8>(tree);
      const vector<PARENT_NODE_IDX_T> & parents=get<9>(tree);
      const vector<SWC_TYPE_T> & swc_types=get<10>(tree);

      throw_assert_nomsg(src_vector.size() > 0);
      throw_assert_nomsg(dst_vector.size() > 0);
      throw_assert_nomsg(sections.size() > 0);
      
      size_t num_xpoints = xcoords.size();
      size_t num_ypoints = ycoords.size();
      size_t num_zpoints = zcoords.size();

      throw_assert_nomsg(num_xpoints == num_ypoints);
      throw_assert_nomsg(num_xpoints == num_zpoints);
      throw_assert_nomsg(num_xpoints == radiuses.size());
      throw_assert_nomsg(num_xpoints == layers.size());
      throw_assert_nomsg(num_xpoints == parents.size());
      throw_assert_nomsg(num_xpoints == swc_types.size());

      size_t num_sections = sections[0];

      size_t num_nodes = num_xpoints, sections_ptr=1;

      set<NODE_IDX_T> all_section_nodes;
      
      while (sections_ptr < sections.size())
        {
          std::vector<NODE_IDX_T> section_nodes;
          size_t num_section_nodes = sections[sections_ptr];
          sections_ptr++;
          for (size_t p = 0; p < num_section_nodes; p++)
            {
                        
              NODE_IDX_T node_idx = sections[sections_ptr];
              if (!(node_idx <= num_nodes))
                {
                  printf("tree id = %u\n",tree_id);
                  printf("node_idx = %u\n",node_idx);
                  printf("num_nodes = %lu\n", num_nodes);
                }
              throw_assert_nomsg(node_idx <= num_nodes);
              all_section_nodes.insert(node_idx);
              sections_ptr++;
            }
      
        }
  
      throw_assert_nomsg(all_section_nodes.size() == num_nodes);

      Graph S;

      for (size_t s = 0; s < num_sections; s++)
        {
          S.insert_vertex(s);
        }
      for (size_t e = 0; e < src_vector.size(); e++)
        {
          S.insert_edge(src_vector[e], dst_vector[e]);
        }

      size_t root_count = 0;
      for (Graph::const_iterator p=S.begin(); p != S.end(); p++)
          {
            Graph::vertex v = Graph::node (p); 
            Graph::vertex_set in = S.in_neighbors(v);
            if (in.size() == 0)
              {
                root_count++;
              }
            
          }

      throw_assert(root_count == 1, "tree must have only one root");
    }

  }
}
