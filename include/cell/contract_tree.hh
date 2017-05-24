// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file contract_tree.hh
///
///  Definition for tree contraction routine.
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================
#ifndef CONTRACT_TREE_HH
#define CONTRACT_TREE_HH

#include <vector>

#include "neurotrees_types.hh"
#include "ngraph.hh"

namespace neurotrees
{
  void contract_tree_bfs (const NGraph::Graph &A, 
                          NGraph::Graph::vertex_set& roots,
                          NGraph::Graph &S, neurotrees::contraction_map_t& contraction_map,
                          NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
  void contract_tree_dfs (const NGraph::Graph &A, 
                          NGraph::Graph::vertex_set& roots,
                          NGraph::Graph &S, neurotrees::contraction_map_t& contraction_map,
                          NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
  void contract_tree_regions_bfs (const NGraph::Graph &A, const std::vector<LAYER_IDX_T>& regions,
                                  NGraph::Graph::vertex_set& roots,
                                  NGraph::Graph &S, neurotrees::contraction_map_t& contraction_map,
                                  NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
  void contract_tree_regions_dfs (const NGraph::Graph &A, const std::vector<LAYER_IDX_T>& regions,
                                  NGraph::Graph::vertex_set& roots,
                                  NGraph::Graph &S, neurotrees::contraction_map_t& contraction_map,
                                  NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
}

#endif
