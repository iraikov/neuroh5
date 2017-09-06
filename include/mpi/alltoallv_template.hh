
#include <mpi.h>

#include <cassert>
#include <vector>
#include <map>

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
      assert(MPI_Comm_size(comm, &ssize) >= 0);

      assert(ssize > 0);
      size = ssize;

      
    /***************************************************************************
     * Send MPI packed data with Alltoallv 
     **************************************************************************/
      recvcounts.resize(size,0);
      rdispls.resize(size,0);
      
      // 1. Each ALL_COMM rank sends a data size to every other rank and
      //    creates sendcounts and sdispls arrays

      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                          &recvcounts[0], 1, MPI_INT, comm) >= 0);
    
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
    
      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], datatype,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], datatype,
                           comm) >= 0);

      return 0;
    }
  }
}
