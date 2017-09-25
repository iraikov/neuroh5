// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file contract_tree.hh
///
///  Definition for tree contraction routine.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================
#ifndef CONTRACT_TREE_HH
#define CONTRACT_TREE_HH

#include <vector>

#include "neuroh5_types.hh"
#include "ngraph.hh"

namespace neuroh5
{
  namespace cell
  {
    void contract_tree_bfs (const NGraph::Graph &A,
                            const std::vector<SWC_TYPE_T>& types,
                            NGraph::Graph::vertex_set& roots,
                            NGraph::Graph &S, contraction_map_t& contraction_map,
                            NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
    void contract_tree_dfs (const NGraph::Graph &A, 
                            const std::vector<SWC_TYPE_T>& types,
                            NGraph::Graph::vertex_set& roots,
                            NGraph::Graph &S, contraction_map_t& contraction_map,
                            NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
    void contract_tree_regions_bfs (const NGraph::Graph &A,
                                    const std::vector<LAYER_IDX_T>& regions,
                                    const std::vector<SWC_TYPE_T>& types,
                                    NGraph::Graph::vertex_set& roots,
                                    NGraph::Graph &S, contraction_map_t& contraction_map,
                                    NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
    void contract_tree_regions_dfs (const NGraph::Graph &A,
                                    const std::vector<LAYER_IDX_T>& regions,
                                    const std::vector<SWC_TYPE_T>& types,
                                    NGraph::Graph::vertex_set& roots,
                                    NGraph::Graph &S, contraction_map_t& contraction_map,
                                    NGraph::Graph::vertex sp, NGraph::Graph::vertex spp);
  }
}

#endif
