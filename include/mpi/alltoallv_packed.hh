#ifndef ALLTOALLV_PACKED_HH
#define ALLTOALLV_PACKED_HH

#include <mpi.h>

#include <vector>
#include <map>

namespace neuroh5
{

  namespace mpi
  {
    int alltoallv_packed (MPI_Comm comm,
                          const vector<int>& sendcounts,
                          const vector<int>& sdispls,
                          const vector<uint8_t>& sendbuf,
                          vector<int>& recvcounts,
                          vector<int>& rdispls,
                          vector<uint8_t>& recvbuf);
  }
}

#endif
