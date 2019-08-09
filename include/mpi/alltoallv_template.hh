// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file alltoallv_template.hh
///
///  Function for sending data via MPI Alltoallv.
///
///  Copyright (C) 2017-2019 Project NeuroH5.
//==============================================================================

#include <mpi.h>

#include <vector>
#include <map>

#include "mpi_debug.hh"
#include "throw_assert.hh"
#include "neuroh5_types.hh"
#include "attr_map.hh"

using namespace std;


namespace neuroh5
{

  namespace mpi
  {

    template<class T>
    int alltoallv_vector (MPI_Comm comm,
                          const MPI_Datatype datatype,
                          const vector<int>& sendcounts,
                          const vector<int>& sdispls,
                          const vector<T>& sendbuf,
                          vector<int>& recvcounts,
                          vector<int>& rdispls,
                          vector<T>& recvbuf)
    {
      int ssize; size_t size;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS,
                   "alltoallv: unable to obtain size of MPI communicator")

      size = ssize;

      
    /***************************************************************************
     * Send MPI packed data with Alltoallv 
     **************************************************************************/
      recvcounts.resize(size,0);
      rdispls.resize(size,0);
      
      // 1. Each ALL_COMM rank sends a data size to every other rank and
      //    creates sendcounts and sdispls arrays

      {
        int status;
        status = MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                              &recvcounts[0], 1, MPI_INT, comm);
        throw_assert(status == MPI_SUCCESS,
                     "alltoallv: error in MPI_Alltoallv: status: " << status);
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

      size_t global_recvbuf_size=0;
      {
        int status;
        status = MPI_Allreduce(&recvbuf_size, &global_recvbuf_size, 1, MPI_SIZE_T, MPI_SUM,
                               comm);
        throw_assert (status == MPI_SUCCESS, "error in MPI_Allreduce: status = " << status);
      }
      if (global_recvbuf_size > 0)
        {
          int status;

          // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
          status = MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], datatype,
                                 &recvbuf[0], &recvcounts[0], &rdispls[0], datatype,
                                 comm);
          throw_assert (status == MPI_SUCCESS, "error in MPI_Alltoallv: status = " << status);
                        
        }

      return 0;
    }
  }
}
