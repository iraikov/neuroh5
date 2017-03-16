#ifndef WRITE_CONNECTIVITY_HH
#define WRITE_CONNECTIVITY_HH

#include <string>
#include <vector>
#include <map>
#include <hdf5.h>

#include "ngh5_types.hh"


namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      void write_projection
      (
       hid_t                     file,
       const std::string&        projection_name,
       const POP_IDX_T&          src_pop_idx,
       const POP_IDX_T&          dst_pop_idx,
       const NODE_IDX_T&         src_start,
       const NODE_IDX_T&         src_end,
       const NODE_IDX_T&         dst_start,
       const NODE_IDX_T&         dst_end,
       const uint64_t&           num_edges,
       const edge_map_t&         prj_edge_map,
       hsize_t            cdim = 4096 
       );

    }
  }
}

#endif
