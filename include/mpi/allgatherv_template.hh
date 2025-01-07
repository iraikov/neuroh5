// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file allgatherv_template.hh
///
///  Function for sending data via MPI Allgatherv.
///
///  Copyright (C) 2017-2024 Project NeuroH5.
//==============================================================================

#ifndef ALLGATHERV_TEMPLATE_HH
#define ALLGATHERV_TEMPLATE_HH

#include <mpi.h>

#include <vector>
#include <map>

#include "mpi_debug.hh"
#include "throw_assert.hh"
#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "chunk_info.hh"

using namespace std;


namespace neuroh5
{

  namespace mpi
  {
    
    template<class T>
    int allgatherv_vector (MPI_Comm comm,
                           const MPI_Datatype datatype,
                           size_t sendcount,
                           const vector<T>& sendbuf,
                           vector<size_t>& recvcounts,
                           vector<size_t>& rdispls,
                           vector<T>& recvbuf)
    {
      int ssize; size_t size;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS,
                   "allgatherv: unable to obtain size of MPI communicator");
      throw_assert_nomsg(ssize > 0);
      size = ssize;

      // Exchange counts

    /***************************************************************************
     * Send MPI data with Allgatherv 
     **************************************************************************/
      recvcounts.resize(size,0);
      rdispls.resize(size,0);
      
      // 1. Each ALL_COMM rank sends a data size to every other rank and
      //    creates sendcounts and sdispls arrays

      {
        int status;
        MPI_Request request;

        status = MPI_Iallgather(&sendcount, 1, MPI_SIZE_T,
                                &recvcounts[0], 1, MPI_SIZE_T, comm,
                                &request);
        throw_assert(status == MPI_SUCCESS,
                     "allgatherv: error in MPI_Iallgather: status: " << status);
        status = MPI_Wait(&request, MPI_STATUS_IGNORE);
        throw_assert(status == MPI_SUCCESS,
                     "allgatherv: error in MPI_Wait: status: " << status);
        
      }
        
      // 2. Each rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      rdispls[0] = 0;
      size_t recvbuf_size = recvcounts[0];
      for (size_t p = 1; p < size; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }

      //assert(recvbuf_size > 0);
      recvbuf.resize(recvbuf_size, 0);

      {
        int status;
        MPI_Request request;

        // 3. Perform the actual data exchange in chunks
        size_t chunk_start = 0;
        while (true)
          {
            size_t global_remaining = 0;
            size_t remaining = 0;
            if (sendcount > chunk_start)
              {
                remaining = sendcount - chunk_start;
              }
            
            status = MPI_Iallreduce(&remaining,
                                    &global_remaining,
                                    1, MPI_SIZE_T, MPI_SUM,
                                    comm, &request);
            
            throw_assert (status == MPI_SUCCESS, "error in MPI_Iallreduce: status = " << status);
            status = MPI_Wait(&request, MPI_STATUS_IGNORE);
            throw_assert(status == MPI_SUCCESS,
                         "allgatherv: error in MPI_Wait: status: " << status);

            if (global_remaining == 0)
              break;
            
            size_t current_chunk = (chunk_start < sendcount) ? std::min(data::CHUNK_SIZE, sendcount - chunk_start) : 0;

            std::vector<int> chunk_recvcounts(size);
            std::vector<int> chunk_displs(size);

            for (rank_t i = 0; i < size; ++i) {
                chunk_recvcounts[i] = static_cast<int>(
                                                       ((chunk_start < recvcounts[i]) ? (recvcounts[i] - chunk_start) : 0));
                chunk_displs[i] = static_cast<int>(rdispls[i] + chunk_start);
            }

            
            status = MPI_Iallgatherv(&sendbuf[chunk_start],
                                     static_cast<int>(current_chunk),
                                     datatype,
                                     &recvbuf[0],
                                     &chunk_recvcounts[0],
                                     &chunk_displs[0],
                                     datatype,
                                     comm, &request);
            throw_assert (status == MPI_SUCCESS, "error in MPI_Alltoallv: status = " << status);
            status = MPI_Wait(&request, MPI_STATUS_IGNORE);
            throw_assert(status == MPI_SUCCESS,
                           "allgatherv: error in MPI_Wait: status: " << status);
            chunk_start = chunk_start + current_chunk;
          }
      }

      return MPI_SUCCESS;
    }
  }
}

#endif
