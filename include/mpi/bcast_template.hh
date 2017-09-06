// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file bcast_string_vector.cc
///
///  Function for sending data via MPI Bcast.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================
#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{

  namespace mpi
  {

    template<class T>
    int bcast_vector (MPI_Comm comm, 
                      const MPI_Datatype datatype,
                      int root,
                      vector< vector<T> > &data)
    {
      int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      uint32_t num_items = data.size();
      assert(MPI_Bcast(&num_items, 1, MPI_UINT32_T, root, comm) >= 0);
      
      if (num_items > 0)
        {
          // allocate buffer
          vector<uint32_t> items_lengths(num_items);
          vector<uint32_t> items_displs(num_items, 0);
          
          uint32_t items_total_length = 0;
          
          // MPI rank 0 reads and broadcasts the population names
          if (rank == root)
            {
              for (size_t i = 0; i < num_items; ++i)
                {
                  items_lengths[i] = data[i].length();
                  items_displs[i]  = data[i].length();
                  items_total_length += items_lengths[i];
                }
            }
          
          // Broadcast string lengths
          assert(MPI_Bcast(&items_total_length, 1, MPI_UINT32_T, root, comm) >= 0);
          assert(MPI_Bcast(&items_lengths[0], num_items, MPI_UINT32_T, root, comm) >= 0);

          for (size_t i = 1; i < num_items; ++i)
            {
              items_displs[i] = items_lengths[i-1];
            }
          
          // Broadcast strings
          size_t offset = 0;
          vector<T> sendbuf;
          
          if (rank == root)
            {
              for (size_t i = 0; i < num_items; i++)
                {
                  copy(data[i].start(), data[i].end(), back_inserter(sendbuf));
                }
            }

          assert(sendbuf.size() = items_total_length);
          assert(MPI_Bcast(sendbuf, sendbuf.size(), datatype, root, comm) >= 0);
        }
      
      return 0;
    }

  }
}
