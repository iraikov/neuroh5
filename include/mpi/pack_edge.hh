// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file pack_edge.cc
///
///  Function for packing edges in MPI_PACKED format.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================

#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>

#include "model_types.hh"
#include "read_graph.hh"

namespace ngh5
{

  namespace mpi
  {

    void pack_edge_map (MPI_Comm comm,
                        MPI_Datatype header_type,
                        MPI_Datatype size_type,
                        model::edge_map_t& prj_edge_map, 
                        size_t &num_packed_edges,
                        int &sendpos,
                        std::vector<uint8_t> &sendbuf);


    void unpack_edge_map (MPI_Comm comm,
                          MPI_Datatype header_type,
                          MPI_Datatype size_type,
                          const std::vector<uint8_t> &recvbuf,
                          const std::vector<uint32_t> &edge_attr_num,
                          model::edge_map_t& prj_edge_map
                          );
    
    void pack_rank_edge_map (MPI_Comm comm,
                             MPI_Datatype header_type,
                             MPI_Datatype size_type,
                             model::rank_edge_map_t& prj_rank_edge_map,
                             size_t &num_packed_edges,
                             std::vector<int>& sendcounts,
                             std::vector<uint8_t> &sendbuf,
                             std::vector<int> &sdispls
                             );

    void unpack_rank_edge_map (MPI_Comm comm,
                               MPI_Datatype header_type,
                               MPI_Datatype size_type,
                               const size_t io_size,
                               const std::vector<uint8_t> &recvbuf,
                               const std::vector<int>& recvcounts,
                               const std::vector<int>& rdispls,
                               const std::vector<uint32_t> &edge_attr_num,
                               model::edge_map_t& prj_edge_map
                               );
  }
}
