#ifndef WRITE_CONNECTIVITY_HH
#define WRITE_CONNECTIVITY_HH

#include <string>
#include <vector>
#include <map>
#include <hdf5.h>

#include "neuroio_types.hh"

namespace neuroio
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
     const model::edge_map_t&         prj_edge_map,
     const std::vector<std::vector<std::string>>& edge_attr_names,
     hsize_t            cdim = 4096 
     );


  }
}

#endif
