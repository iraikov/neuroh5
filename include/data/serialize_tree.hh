// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file serialize_tree.hh
///
///  Functions for serializing tree data.
///
///  Copyright (C) 2017-2021 Project Neuroh5.
//==============================================================================

#ifndef SERIALIZE_TREE_HH
#define SERIALIZE_TREE_HH

#include <mpi.h>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <forward_list>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace data
  {
    void serialize_rank_tree_map (const size_t num_ranks,
                                  const size_t start_rank,
                                  const std::map <rank_t, std::map<CELL_IDX_T, neurotree_t> >& rank_tree_map,
                                  std::vector<int>& sendcounts,
                                  std::vector<char> &sendbuf,
                                  std::vector<int> &sdispls);

    void deserialize_rank_tree_map (const size_t num_ranks,
                                    const std::vector<char> &recvbuf,
                                    const std::vector<int>& recvcounts,
                                    const std::vector<int>& rdispls,
                                    std::map<CELL_IDX_T, neurotree_t> &all_tree_map
                                    );

    void deserialize_rank_tree_list (const size_t num_ranks,
                                     const vector<char> &recvbuf,
                                     const vector<int>& recvcounts,
                                     const vector<int>& rdispls,
                                     forward_list<neurotree_t> &all_tree_list);
  }
}
#endif

