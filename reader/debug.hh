#ifndef NGH5DEBUG_HH
#define NGH5DEBUG_HH

#include <cassert>
#include <cstdio>
#include <iostream>

static bool debug_enabled = true;

inline void DEBUG(){}

template<typename First, typename ...Rest>
inline void DEBUG(First && first, Rest && ...rest)
{
  if (debug_enabled)
    {
      std::cerr << std::forward<First>(first);
      DEBUG(std::forward<Rest>(rest)...);
      std::cerr << std::flush;
    }
}
#endif
