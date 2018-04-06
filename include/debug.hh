#ifndef NGH5DEBUG_HH
#define NGH5DEBUG_HH

#include <cassert>
#include <cstdio>
#include <iostream>
#include <mpi.h>

namespace neuroh5
{

  static bool debug_enabled = false;
  
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
