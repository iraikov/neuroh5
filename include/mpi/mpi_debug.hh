// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file mpi_debug.hh
///
///  MPI-specific debugging routines.
///
///  Copyright (C) 2017 Project Neurograph.
//==============================================================================

#ifndef MPI_DEBUG_HH
#define MPI_DEBUG_HH

#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>

#include <debug.hh>
#include <mpe_seq.hh>


#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{

  namespace mpi
  {
  
    template<typename First, typename ...Rest>
    inline void MPI_DEBUG(MPI_Comm comm, First && first, Rest && ...rest)
    {
      if (debug_enabled)
        {
          MPE_Seq_begin( comm, 1 );
          int rank;
          assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
          std::cerr << "Rank " << rank << ": ";
          std::cerr << std::forward<First>(first);
          DEBUG(std::forward<Rest>(rest)...);
          std::cerr << std::endl;
          std::cerr << std::flush;
          MPE_Seq_end( comm, 1 );
        }
    }

  }
}

#endif
