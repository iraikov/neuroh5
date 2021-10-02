// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file insert_tree_points.cc
///
///  Insert points into tree structure.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================


#include "debug.hh"

#include <algorithm>
#include <set>

#include "neuroh5_types.hh"
#include "throw_assert.hh"


using namespace std;
namespace neuroh5
{

  namespace cell
  {
    
    void insert_tree_points(const neurotree_t& src_tree,
                            neurotree_t& dst_tree,
                            LAYER_IDX_T include_layer,
                            bool translate)
    {
      CELL_IDX_T tree_id = get<0>(dst_tree);
      std::deque<SECTION_IDX_T> & src_vector=get<1>(dst_tree);
      std::deque<SECTION_IDX_T> & dst_vector=get<2>(dst_tree);
      std::deque<SECTION_IDX_T> & sections=get<3>(dst_tree);
      std::deque<COORD_T> & xcoords=get<4>(dst_tree);
      std::deque<COORD_T> & ycoords=get<5>(dst_tree);
      std::deque<COORD_T> & zcoords=get<6>(dst_tree);
      std::deque<REALVAL_T> & radiuses=get<7>(dst_tree);
      std::deque<LAYER_IDX_T> & layers=get<8>(dst_tree);
      std::deque<PARENT_NODE_IDX_T> & parents=get<9>(dst_tree);
      std::deque<SWC_TYPE_T> & swc_types=get<10>(dst_tree);

      std::deque<SECTION_IDX_T> include_src_vector=get<1>(src_tree);
      std::deque<SECTION_IDX_T> include_dst_vector=get<2>(src_tree);
      std::deque<SECTION_IDX_T> include_sections=get<3>(src_tree);
      std::deque<COORD_T> include_xcoords=get<4>(src_tree);
      std::deque<COORD_T> include_ycoords=get<5>(src_tree);
      std::deque<COORD_T> include_zcoords=get<6>(src_tree);
      std::deque<REALVAL_T> include_radiuses=get<7>(src_tree);
      std::deque<LAYER_IDX_T> include_layers=get<8>(src_tree);
      std::deque<PARENT_NODE_IDX_T> include_parents=get<9>(src_tree);
      std::deque<SWC_TYPE_T> include_swc_types=get<10>(src_tree);


      size_t num_xpoints = xcoords.size();
      throw_assert_nomsg(num_xpoints > 0);
      
      COORD_T dst_origin_x = xcoords[0];
      COORD_T dst_origin_y = ycoords[0];
      COORD_T dst_origin_z = zcoords[0];
      
      size_t num_nodes = num_xpoints, sections_ptr=1;
      size_t num_sections = sections[0];
      size_t include_num_sections = include_sections[0];

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

      for (auto it = include_src_vector.begin();
           it != include_src_vector.end();
           ++it)
        {
          *it = *it + num_sections;
        }
      for (auto it = include_dst_vector.begin();
           it != include_dst_vector.end();
           ++it)
        {
          *it = *it + num_sections;
        }

      if (translate)
        {
          COORD_T x_offset = dst_origin_x - include_xcoords[0];
          COORD_T y_offset = dst_origin_y - include_ycoords[0];
          COORD_T z_offset = dst_origin_z - include_zcoords[0];
          for (size_t i = 0; i<include_xcoords.size(); i++)
            {
              include_xcoords[i] += x_offset;
              include_ycoords[i] += y_offset;
              include_zcoords[i] += z_offset;
            }
        }
      
      sections_ptr = 1;
      size_t include_num_nodes = include_xcoords.size();
      set<NODE_IDX_T> include_all_section_nodes;
      while (sections_ptr < include_sections.size())
        {
          std::vector<NODE_IDX_T> section_nodes;
          size_t num_section_nodes = include_sections[sections_ptr];
          sections_ptr++;
          for (size_t p = 0; p < num_section_nodes; p++)
            {
              NODE_IDX_T node_idx = include_sections[sections_ptr];
              if (!(node_idx <= include_num_nodes))
                {
                  printf("node_idx = %u\n",node_idx);
                  printf("include_num_nodes = %lu\n", include_num_nodes);
                }
              throw_assert_nomsg(node_idx <= include_num_nodes);
              include_all_section_nodes.insert(node_idx);
              include_sections[sections_ptr] += num_nodes;              
              sections_ptr++;
            }
      
        }
      throw_assert_nomsg(include_all_section_nodes.size() == include_num_nodes);
      
      for (auto it = include_parents.begin();
           it != include_parents.end();
           ++it)
        {
          if (*it > -1)
            {
              *it = *it + num_nodes;
            }
        }

      for (auto it = include_layers.begin();
           it != include_layers.end();
           ++it)
        {
          *it = include_layer;
        }
      
      sections[0] += include_num_sections;
      copy(include_src_vector.begin(), include_src_vector.end(), back_inserter(src_vector));
      copy(include_dst_vector.begin(), include_dst_vector.end(), back_inserter(dst_vector));
      copy(include_sections.begin()+1, include_sections.end(), back_inserter(sections));
      copy(include_xcoords.begin(), include_xcoords.end(), back_inserter(xcoords));
      copy(include_ycoords.begin(), include_ycoords.end(), back_inserter(ycoords));
      copy(include_zcoords.begin(), include_zcoords.end(), back_inserter(zcoords));
      copy(include_radiuses.begin(), include_radiuses.end(), back_inserter(radiuses));
      copy(include_layers.begin(), include_layers.end(), back_inserter(layers));
      copy(include_parents.begin(), include_parents.end(), back_inserter(parents));
      copy(include_swc_types.begin(), include_swc_types.end(), back_inserter(swc_types));
    }

  }
}
