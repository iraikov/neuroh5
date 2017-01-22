#ifndef WRITE_CONNECTIVITY_HH
#define WRITE_CONNECTIVITY_HH

#include "ngh5_types.hh"

#include "hdf5.h"

#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      void write_connectivity
      (
       hid_t                          file,
       const NODE_IDX_T&              first_node,
       const NODE_IDX_T&              last_node,
       const std::vector<NODE_IDX_T>& edges
       );
    }
  }
}

#endif
