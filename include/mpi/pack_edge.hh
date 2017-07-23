// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file pack_edge.hh
///
///  Functions for packing edges in MPI_PACKED format.
///
///  Copyright (C) 2017 Project Neuroh5.
//==============================================================================

#include <mpi.h>
#include <cstring>
#include <vector>
#include <map>
#include <string>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace mpi
  {
    void pack_adj_map (MPI_Comm comm, MPI_Datatype header_type, MPI_Datatype size_type,
                       const std::map<NODE_IDX_T, std::vector<NODE_IDX_T> >& adj_map,
                       size_t &num_packed_edges,
                       int &sendpos,
                       std::vector<uint8_t> &sendbuf);
    
    void pack_edge_map (MPI_Comm comm,
                        MPI_Datatype header_type,
                        MPI_Datatype size_type,
                        edge_map_t& prj_edge_map, 
                        size_t &num_packed_edges,
                        int &sendpos,
                        std::vector<uint8_t> &sendbuf);


    void unpack_edge_map (MPI_Comm comm,
                          MPI_Datatype header_type,
                          MPI_Datatype size_type,
                          const std::vector<uint8_t> &recvbuf,
                          const std::vector<uint32_t> &edge_attr_num,
                          edge_map_t& prj_edge_map
                          );
    
    void pack_rank_edge_map (MPI_Comm comm,
                             MPI_Datatype header_type,
                             MPI_Datatype size_type,
                             rank_edge_map_t& prj_rank_edge_map,
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
                               edge_map_t& prj_edge_map,
                               uint64_t& num_unpacked_edges
                               );
  }
}
