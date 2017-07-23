// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file bcast_string_vector.cc
///
///  Function for broadcasting a string vector via MPI.
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

    int bcast_string_vector (MPI_Comm comm, int root,
                             const size_t max_string_len,
                             vector<string> &string_vector)
    {
      int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      uint32_t num_strings = string_vector.size();
      assert(MPI_Bcast(&num_strings, 1, MPI_UINT32_T, root, comm) >= 0);
    
      // allocate buffer
      vector<uint32_t> string_lengths(num_strings);
      string_vector.resize(num_strings);

      uint32_t strings_total_length = 0;
    
      // MPI rank 0 reads and broadcasts the population names
      if (rank == root)
        {
          for (size_t i = 0; i < num_strings; ++i)
            {
              string_lengths[i] = string_vector[i].length();
              strings_total_length += string_lengths[i];
            }
        }

      // Broadcast string lengths
      assert(MPI_Bcast(&strings_total_length, 1, MPI_UINT32_T, root, comm) >= 0);
      assert(MPI_Bcast(&string_lengths[0], num_strings, MPI_UINT32_T, root, comm) >= 0);

      // Broadcast strings
      size_t offset = 0;
      char* strings_buf = new char [strings_total_length];
      assert(strings_buf != NULL);

      if (rank == root)
        {
          for (size_t i = 0; i < num_strings; i++)
            {
              memcpy(strings_buf+offset, string_vector[i].c_str(),
                     string_lengths[i]);
              offset = offset + string_lengths[i];
            }
        }
    
      assert(MPI_Bcast(strings_buf, strings_total_length, MPI_BYTE, root, comm) >= 0);
    
      // Copy population names into pop_names
      char buf[max_string_len];
      offset = 0;
      for (size_t i = 0; i < num_strings; i++)
        {
          size_t len = string_lengths[i];
          memcpy(buf, strings_buf+offset, len);
          buf[len] = '\0';
          string_vector[i] = string((const char*)buf);
          offset = offset + len;
        }
    
      delete [] strings_buf;
    
      return 0;
    }

  }
}
