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
#include <cstring>
#include <cstdio>

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
      int myrank;
      MPI_Comm_rank(comm, &myrank);


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

      recvbuf.resize(recvbuf_size, 0);

      {
        // 3. Perform the actual data exchange in chunks using point-to-point
        //    sends/receives.
        //
        // Rationale: MPI_Alltoallv uses int displacements, which overflow for
        // large datasets (> ~2 GB per rank pair).  On GPU nodes the resulting
        // negative displacement causes the CMA (process_vm_readv) transport to
        // read from an invalid address, producing "process_vm_readv: Bad address"
        // and SIGABRT/SIGSEGV.  Explicit point-to-point avoids the int cast and
        // uses size_t pointer arithmetic throughout.
        //
        // NEUROH5_CHUNK_SIZE (bytes) caps the maximum send per rank per round.
        // Default 1 GB; set smaller to limit per-message size.
        size_t chunk_size = data::get_chunk_size();
        const int mpi_tag = 9876; // arbitrary fixed tag for alltoallv exchange

        for (size_t chunk_start = 0; ; chunk_start += chunk_size)
          {
            // Compute how many elements each rank sends / receives this round
            std::vector<size_t> send_this(size, 0), recv_this(size, 0);
            size_t global_send = 0, global_recv = 0;
            for (size_t i = 0; i < size; ++i) {
              if (chunk_start < sendcounts[i])
                send_this[i] = std::min(sendcounts[i] - chunk_start, chunk_size);
              if (chunk_start < recvcounts[i])
                recv_this[i] = std::min(recvcounts[i] - chunk_start, chunk_size);
              global_send += send_this[i];
              global_recv += recv_this[i];
            }

            // All-reduce to check if any rank has work left
            size_t g_send_sum = 0, g_recv_sum = 0;
            {
              MPI_Request req;
              MPI_Iallreduce(&global_send, &g_send_sum, 1, MPI_SIZE_T, MPI_SUM, comm, &req);
              MPI_Wait(&req, MPI_STATUS_IGNORE);
              MPI_Iallreduce(&global_recv, &g_recv_sum, 1, MPI_SIZE_T, MPI_SUM, comm, &req);
              MPI_Wait(&req, MPI_STATUS_IGNORE);
            }
            if (g_send_sum == 0 && g_recv_sum == 0)
              break;

            // Self-copy first (no MPI needed for rank-to-self)
            if (send_this[myrank] > 0 && recv_this[myrank] > 0) {
              size_t n = std::min(send_this[myrank], recv_this[myrank]);
              std::memcpy(&recvbuf[rdispls[myrank] + chunk_start],
                          &sendbuf[sdispls[myrank] + chunk_start],
                          n * sizeof(T));
            }

            // Post receives first, then sends (standard non-blocking pattern)
            std::vector<MPI_Request> reqs;
            reqs.reserve(2 * size);

            for (size_t i = 0; i < size; ++i) {
              if ((int)i == myrank) continue;
              if (recv_this[i] > 0) {
                MPI_Request r;
                // Use size_t pointer arithmetic — no int displacement overflow
                T* recv_ptr = &recvbuf[rdispls[i] + chunk_start];
                int status = MPI_Irecv(recv_ptr, (int)recv_this[i], datatype,
                                       (int)i, mpi_tag, comm, &r);
                throw_assert(status == MPI_SUCCESS,
                             "alltoallv: error in MPI_Irecv: status: " << status);
                reqs.push_back(r);
              }
            }
            for (size_t i = 0; i < size; ++i) {
              if ((int)i == myrank) continue;
              if (send_this[i] > 0) {
                MPI_Request r;
                // Direct pointer into sendbuf: avoids int displacement overflow
                const T* send_ptr = &sendbuf[sdispls[i] + chunk_start];
                int status = MPI_Isend(const_cast<T*>(send_ptr), (int)send_this[i],
                                       datatype, (int)i, mpi_tag, comm, &r);
                throw_assert(status == MPI_SUCCESS,
                             "alltoallv: error in MPI_Isend: status: " << status);
                reqs.push_back(r);
              }
            }

            if (!reqs.empty()) {
              int status = MPI_Waitall((int)reqs.size(), &reqs[0], MPI_STATUSES_IGNORE);
              throw_assert(status == MPI_SUCCESS,
                           "alltoallv: error in MPI_Waitall: status: " << status);
            }
          }
      }

      return MPI_SUCCESS;
    }
  }
}

#endif
