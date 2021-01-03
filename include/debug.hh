#ifndef NEUROH5_DEBUG_HH
#define NEUROH5_DEBUG_HH

#include <cassert>
#include <cstdio>
#include <iostream>
#include <mpi.h>

namespace neuroh5
{
#ifdef NEUROH5_DEBUG
  static bool debug_enabled = true;
#else
  static bool debug_enabled = false;
#endif  

  inline void DEBUG(){}
  
  template<typename First, typename ...Rest>
  inline void DEBUG(First && first, Rest && ...rest)
  {
    if (debug_enabled)
      {
        std::cerr << std::forward<First>(first);
        DEBUG(std::forward<Rest>(rest)...);
        std::cerr << std::endl;
        std::cerr << std::flush;
      }
  }

  
  
}
#endif
