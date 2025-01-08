// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file alltoallv_template.hh
///
///  Function for sending data via MPI Alltoallv.
///
///  Copyright (C) 2017-2024 Project NeuroH5.
//==============================================================================

#ifndef ALLTOALLV_TEMPLATE_HH
#define ALLTOALLV_TEMPLATE_HH

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
    int alltoallv_vector (MPI_Comm comm,
                          const MPI_Datatype datatype,
                          const vector<size_t>& sendcounts,
                          const vector<size_t>& sdispls,
                          const vector<T>& sendbuf,
                          vector<size_t>& recvcounts,
                          vector<size_t>& rdispls,
                          vector<T>& recvbuf)
    {
      int ssize; size_t size;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS,
                   "alltoallv: unable to obtain size of MPI communicator");
      throw_assert_nomsg(ssize > 0);
      size = ssize;

      
    /***************************************************************************
     * Send MPI data with Alltoallv 
     **************************************************************************/
      recvcounts.resize(size,0);
      rdispls.resize(size,0);
      
      // 1. Each ALL_COMM rank sends a data size to every other rank and
      //    creates sendcounts and sdispls arrays

      {
        int status;
        MPI_Request request;
        status = MPI_Ialltoall(&sendcounts[0], 1, MPI_SIZE_T,
                               &recvcounts[0], 1, MPI_SIZE_T,
                               comm, &request);
        throw_assert(status == MPI_SUCCESS,
                     "alltoallv: error in MPI_Ialltoall: status: " << status);
        status = MPI_Wait(&request, MPI_STATUS_IGNORE);
        throw_assert(status == MPI_SUCCESS,
                     "alltoallv: error in MPI_Wait: status: " << status);
        
      }
        
      // 2. Each rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      
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
            size_t global_send_size=0;
            size_t global_recv_size=0;
            
            auto chunk = data::calculate_chunk_sizes<T>(
                sendcounts, sdispls, recvcounts, rdispls,
                chunk_start, data::CHUNK_SIZE);

            status = MPI_Iallreduce(&chunk.total_recv_size,
                                    &global_recv_size,
                                    1, MPI_SIZE_T, MPI_SUM,
                                    comm, &request);
            
            throw_assert (status == MPI_SUCCESS, "error in MPI_Iallreduce: status = " << status);
            status = MPI_Wait(&request, MPI_STATUS_IGNORE);
            throw_assert(status == MPI_SUCCESS,
                         "alltoallv: error in MPI_Wait: status: " << status);

            status = MPI_Iallreduce(&chunk.total_send_size,
                                    &global_send_size,
                                    1, MPI_SIZE_T, MPI_SUM,
                                    comm, &request);
            
            throw_assert (status == MPI_SUCCESS, "error in MPI_Iallreduce: status = " << status);
            status = MPI_Wait(&request, MPI_STATUS_IGNORE);
            throw_assert(status == MPI_SUCCESS,
                         "alltoallv: error in MPI_Wait: status: " << status);

            if (global_send_size == 0 &&
                global_recv_size == 0)
              break;
            
            status = MPI_Ialltoallv(&sendbuf[0],
                                    &chunk.sendcounts[0],
                                    &chunk.sdispls[0],
                                    datatype,
                                    &recvbuf[0],
                                    &chunk.recvcounts[0],
                                    &chunk.rdispls[0],
                                    datatype,
                                    comm, &request);
            throw_assert (status == MPI_SUCCESS, "error in MPI_Alltoallv: status = " << status);
            status = MPI_Wait(&request, MPI_STATUS_IGNORE);
            throw_assert(status == MPI_SUCCESS,
                           "alltoallv: error in MPI_Wait: status: " << status);

            chunk_start += data::CHUNK_SIZE;

          }
      }

      return MPI_SUCCESS;
    }
  }
}

#endif
